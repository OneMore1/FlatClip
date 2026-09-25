from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OUT_BASE = Path("<path>")
LABEL_DIR = Path("<path>")
STIM_INFO = Path("<path>")
RESULTS_DIR = Path("<path>")
CKPT_DIR = Path("<path>")
SUBJECTS = ("sub1", "sub2", "sub5", "sub7")
SUBJECT_COL = {
    "sub1": "subject1",
    "sub2": "subject2",
    "sub5": "subject5",
    "sub7": "subject7",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DINO image-token classifier for COCO80 with per-subject within/loso splits."
    )
    p.add_argument("--cache-base", type=Path, default=OUT_BASE)
    p.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    p.add_argument("--stim-info", type=Path, default=STIM_INFO)
    p.add_argument("--target-name", type=str, required=True)
    p.add_argument("--token-mode", choices=("full", "last"), required=True)
    p.add_argument("--regime", choices=("within", "loso"), required=True)
    p.add_argument("--test-subject", choices=SUBJECTS, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--best-metric", choices=("mAP", "micro_f1", "weighted_f1"), default="mAP")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool_cell(value: object) -> bool:
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", "", "nan", "none"}:
        return False
    try:
        return float(s) > 0
    except ValueError:
        return False


def load_subject_membership(stim_info: Path) -> tuple[set[int], dict[str, set[int]]]:
    shared: set[int] = set()
    subject_ids: dict[str, set[int]] = {s: set() for s in SUBJECTS}
    with stim_info.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nsd_id = int(row["nsdId"])
            if parse_bool_cell(row.get("shared1000", "0")):
                shared.add(nsd_id)
            for subject, col in SUBJECT_COL.items():
                if parse_bool_cell(row.get(col, "0")):
                    subject_ids[subject].add(nsd_id)
    return shared, subject_ids


def json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class TokenDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, nsd_ids: np.ndarray):
        self.x = x
        self.y = y.astype(np.float32, copy=False)
        self.nsd_ids = nsd_ids.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.nsd_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.asarray(self.x[idx], dtype=np.float32)),
            torch.from_numpy(self.y[idx].copy()),
            torch.tensor(int(self.nsd_ids[idx]), dtype=torch.long),
        )


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DinoTokenClassifier(nn.Module):
    def __init__(
        self,
        token_mode: str,
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.1,
        out_dim: int = 80,
    ):
        super().__init__()
        self.token_mode = token_mode
        in_dim = 768 if token_mode == "last" else 768 * 3  # noqa: S105 - model token mode
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.head = nn.Linear(hidden_dim, out_dim)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_mode == "last":  # noqa: S105 - model token mode
            return x[:, -1, :]
        cls = x[:, 0, :]
        patches = x[:, 1:, :]
        patch_mean = patches.mean(dim=1)
        patch_max = patches.amax(dim=1)
        return torch.cat([cls, patch_mean, patch_max], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_tokens(x)
        h = self.stem(h)
        h = self.blocks(h)
        return self.head(h)


def average_precision_binary(target: np.ndarray, score: np.ndarray) -> float:
    pos = int(target.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = target[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y), dtype=np.float64) + 1.0)
    return float((precision * y).sum() / pos)


def evaluate_multilabel_arrays(logits: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    aps = []
    for cls in range(target.shape[1]):
        ap = average_precision_binary(target[:, cls], probs[:, cls])
        if not np.isnan(ap):
            aps.append(ap)
    mAP = float(np.mean(aps)) if aps else 0.0

    pred = probs >= threshold
    tgt = target.astype(bool)
    tp = np.logical_and(pred, tgt).sum(axis=0).astype(np.float64)
    fp = np.logical_and(pred, np.logical_not(tgt)).sum(axis=0).astype(np.float64)
    fn = np.logical_and(np.logical_not(pred), tgt).sum(axis=0).astype(np.float64)
    support = tgt.sum(axis=0).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1_per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_f1 = float(np.mean(f1_per_class))
    total_support = support.sum()
    weighted_f1 = float((f1_per_class * support).sum() / total_support) if total_support > 0 else 0.0

    tp_micro = float(tp.sum())
    fp_micro = float(fp.sum())
    fn_micro = float(fn.sum())
    denom_micro = 2 * tp_micro + fp_micro + fn_micro
    micro_f1 = float((2 * tp_micro / denom_micro) if denom_micro > 0 else 0.0)

    subset_acc = float((pred == tgt).all(axis=1).mean())
    label_acc = float((pred == tgt).mean())

    return {
        "mAP": mAP,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "subset_acc": subset_acc,
        "label_acc": label_acc,
        "positive_rate": float(tgt.mean()),
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    logits_list = []
    target_list = []
    ids_list = []
    for x, y, batch_ids in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="mean")
        b = int(x.shape[0])
        total_loss += float(loss.item()) * b
        total += b
        logits_list.append(logits.float().cpu().numpy())
        target_list.append(y.float().cpu().numpy())
        ids_list.append(batch_ids.numpy())
    logits_all = np.concatenate(logits_list, axis=0)
    target_all = np.concatenate(target_list, axis=0)
    ids_all = np.concatenate(ids_list, axis=0)
    metrics = evaluate_multilabel_arrays(logits_all, target_all, threshold=threshold)
    metrics["bce"] = total_loss / max(total, 1)
    metrics["n"] = int(total)
    return metrics, logits_all, target_all, ids_all


def build_split(
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    shared, subject_ids = load_subject_membership(args.stim_info)
    token_root = args.cache_base / "dino_targets" / args.target_name
    token_ids = np.load(token_root / "ids.npy")
    token_x = np.load(token_root / "y_tokens.npy", mmap_mode="r")
    label_ids = np.load(args.label_dir / "ids.npy")
    label_y = np.load(args.label_dir / "y_coco80.npy", mmap_mode="r")

    token_index = {int(nsd_id): idx for idx, nsd_id in enumerate(token_ids.tolist())}
    label_index = {int(nsd_id): idx for idx, nsd_id in enumerate(label_ids.tolist())}

    train_subjects = (
        [args.test_subject]
        if args.regime == "within"
        else [subject for subject in SUBJECTS if subject != args.test_subject]
    )

    train_pool: set[int] = set()
    for subject in train_subjects:
        train_pool.update(subject_ids[subject])
    train_pool.difference_update(shared)
    test_pool = set(subject_ids[args.test_subject]).intersection(shared)

    def to_valid_ids(pool: set[int]) -> np.ndarray:
        vals = [nsd_id for nsd_id in sorted(pool) if nsd_id in token_index and nsd_id in label_index]
        return np.asarray(vals, dtype=np.int64)

    train_ids_all = to_valid_ids(train_pool)
    test_ids = to_valid_ids(test_pool)
    if len(train_ids_all) < 2:
        raise ValueError(f"Insufficient train ids for {args.regime}/{args.test_subject}: {len(train_ids_all)}")
    if len(test_ids) == 0:
        raise ValueError(f"No test ids for {args.regime}/{args.test_subject}")

    rng = np.random.default_rng(args.seed)
    perm = np.arange(len(train_ids_all))
    rng.shuffle(perm)
    n_val = max(1, round(len(perm) * args.val_frac)) if len(perm) > 1 else 0
    n_val = min(n_val, max(0, len(perm) - 1))
    val_sel = perm[:n_val]
    train_sel = perm[n_val:]

    train_ids = train_ids_all[train_sel]
    val_ids = train_ids_all[val_sel]

    def gather_x(ids: np.ndarray) -> np.ndarray:
        idx = np.asarray([token_index[int(nsd_id)] for nsd_id in ids.tolist()], dtype=np.int64)
        return np.asarray(token_x[idx], dtype=np.float16)

    def gather_y(ids: np.ndarray) -> np.ndarray:
        idx = np.asarray([label_index[int(nsd_id)] for nsd_id in ids.tolist()], dtype=np.int64)
        return np.asarray(label_y[idx], dtype=np.float32)

    return (
        gather_x(train_ids),
        gather_y(train_ids),
        train_ids,
        gather_x(val_ids),
        gather_y(val_ids),
        val_ids,
        gather_x(test_ids),
        gather_y(test_ids),
        test_ids,
        train_subjects,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    (
        train_x,
        train_y,
        train_ids,
        val_x,
        val_y,
        val_ids,
        test_x,
        test_y,
        test_ids,
        train_subjects,
    ) = build_split(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    train_ds = TokenDataset(train_x, train_y, train_ids)
    val_ds = TokenDataset(val_x, val_y, val_ids)
    test_ds = TokenDataset(test_x, test_y, test_ids)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DinoTokenClassifier(
        token_mode=args.token_mode,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        out_dim=80,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    pos = torch.from_numpy(train_y.sum(axis=0)).float()
    neg = float(len(train_y)) - pos
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), min=1.0, max=50.0).to(device)

    tag = (
        f"nsd_image_dino_coco80_{args.target_name}_{args.token_mode}_{args.regime}_{args.test_subject}_seed{args.seed}"
    )
    out = args.out or (RESULTS_DIR / f"{tag}.json")
    ckpt = args.ckpt or (CKPT_DIR / f"{tag}.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] target={args.target_name} token_mode={args.token_mode} regime={args.regime} "
        f"test_subject={args.test_subject} train_subjects={train_subjects} train={len(train_ds)} "
        f"val={len(val_ds)} test={len(test_ds)} token_shape={tuple(train_x.shape[1:])} "
        f"params={n_params:,} device={device} amp={amp}",
        flush=True,
    )

    history = []
    best_val = None
    best_state = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        epoch_start = time.time()
        for step, (x, y, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=pos_weight)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            b = int(x.shape[0])
            running += float(loss.item()) * b
            seen += b
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={step}/{len(train_loader)} loss={running / max(seen, 1):.6f}",
                    flush=True,
                )
        train_loss = running / max(seen, 1)
        val_metrics, _, _, _ = evaluate_classifier(model, val_loader, device, amp)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        print(
            f"[epoch {epoch}] train_loss={train_loss:.6f} val_mAP={val_metrics['mAP']:.4f} "
            f"val_label_acc={val_metrics['label_acc']:.4f} val_weighted_f1={val_metrics['weighted_f1']:.4f}",
            flush=True,
        )
        current = float(val_metrics[args.best_metric])
        if best_val is None or current > best_val:
            best_val = current
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint.")
    model.load_state_dict(best_state)
    test_metrics, _, _, pred_ids = evaluate_classifier(model, test_loader, device, amp)
    payload = {
        "task": "DINO image token -> COCO80 multi-label classification (subject split)",
        "target_name": args.target_name,
        "token_mode": args.token_mode,
        "regime": args.regime,
        "test_subject": args.test_subject,
        "train_subjects": train_subjects,
        "config": vars(args),
        "params": int(n_params),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test_shared1000": len(test_ds),
        "best_val_metric": float(best_val),
        "history": history,
        "test_metrics": test_metrics,
        "test_nsd_ids": pred_ids.astype(int).tolist(),
        "train_seconds": time.time() - start,
    }
    json_dump(out, payload)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "payload": payload}, ckpt)
    print(json.dumps({"test_metrics": test_metrics}, indent=2), flush=True)
    print(f"[saved] out={out} ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
