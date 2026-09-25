from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from inference.nsd_utils import (
    CKPT_DIR,
    LABEL_DIR,
    RESULTS_DIR,
    STIM_INFO,
    SUBJECTS,
    evaluate_classifier,
    json_dump,
    load_subject_membership,
    set_seed,
)

HDF5_ROOT = Path("<path>")
CACHE_ROOT = Path("<path>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train COCO80 classifiers on nsdgeneral ROI SigLIP2 patch tokens.")
    p.add_argument("--hdf5-root", type=Path, default=HDF5_ROOT)
    p.add_argument("--hdf5-template", type=str, default="{subject}_nsd_roi.hdf5")
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--embedding-label", type=str, default="nsd_roi")
    p.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    p.add_argument("--stim-info", type=Path, default=STIM_INFO)
    p.add_argument("--input-mode", choices=("tokens256", "one_token"), required=True)
    p.add_argument("--one-token-source", choices=("patch_mean", "token_mean"), default="patch_mean")
    p.add_argument("--regime", choices=("within", "loso"), required=True)
    p.add_argument("--test-subject", choices=SUBJECTS, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--token-hidden", type=int, default=128)
    p.add_argument("--channel-hidden", type=int, default=2048)
    p.add_argument("--classifier-hidden", type=int, default=1024)
    p.add_argument("--classifier-depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--best-metric", choices=("mAP", "micro_f1", "weighted_f1"), default="mAP")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def safe_slug(text: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text.strip())
    return slug.strip("._-") or "embedding"


def public_config(args: argparse.Namespace) -> dict[str, object]:
    """Serialize hyperparameters without local data, checkpoint, or output paths."""
    hidden = {
        "hdf5_root",
        "cache_root",
        "label_dir",
        "stim_info",
        "out",
        "ckpt",
    }
    return {key: value for key, value in vars(args).items() if key not in hidden}


def hdf5_path(hdf5_root: Path, subject: str, hdf5_template: str) -> Path:
    return hdf5_root / hdf5_template.format(subject=subject)


def cache_paths(cache_root: Path, subject: str, embedding_label: str) -> dict[str, Path]:
    label = safe_slug(embedding_label)
    if cache_root.name.startswith(f"{label}_"):
        root = cache_root / "siglip2_base_patch16_naflex" / subject
    else:
        root = cache_root / label / "siglip2_base_patch16_naflex" / subject
    return {
        "root": root,
        "ids": root / "ids.npy",
        "tokens": root / "x_tokens256.npy",
        "patch_mean": root / "x_one_token_patch_mean.npy",
        "token_mean": root / "x_one_token_token_mean.npy",
        "metadata": root / "metadata.json",
    }


def cache_is_valid(paths: dict[str, Path]) -> bool:
    required = [paths["ids"], paths["tokens"], paths["patch_mean"], paths["token_mean"]]
    if not all(path.exists() for path in required):
        return False
    try:
        ids = np.load(paths["ids"], mmap_mode="r")
        tokens = np.load(paths["tokens"], mmap_mode="r")
        patch_mean = np.load(paths["patch_mean"], mmap_mode="r")
        token_mean = np.load(paths["token_mean"], mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (
        ids.ndim == 1
        and tokens.shape == (ids.shape[0], 256, 768)
        and tokens.dtype == np.float16
        and patch_mean.shape == (ids.shape[0], 768)
        and token_mean.shape == (ids.shape[0], 768)
    )


def ensure_hdf5_cache(
    *,
    hdf5_root: Path,
    hdf5_template: str,
    cache_root: Path,
    subject: str,
    embedding_label: str,
    force: bool = False,
) -> dict[str, Path]:
    paths = cache_paths(cache_root, subject, embedding_label)
    if not force and cache_is_valid(paths):
        print(
            f"[cache] using existing {embedding_label}_siglip2/{subject}: {paths['root']}",
            flush=True,
        )
        return paths

    src = hdf5_path(hdf5_root, subject, hdf5_template)
    if not src.exists():
        raise FileNotFoundError(src)
    paths["root"].mkdir(parents=True, exist_ok=True)
    tmp = {
        key: path.with_suffix(".tmp.npy")
        for key, path in paths.items()
        if key in {"ids", "tokens", "patch_mean", "token_mean"}
    }
    for path in tmp.values():
        if path.exists():
            path.unlink()

    with h5py.File(src, "r") as h5:
        ids = np.asarray(h5["nsd_ids"][:], dtype=np.int64)
        ds = h5["embeddings"]
        if ds.shape != (len(ids), 256, 768):
            raise ValueError(f"Expected {(len(ids), 256, 768)} in {src}, got {ds.shape}")
        x_tokens = np.lib.format.open_memmap(tmp["tokens"], mode="w+", dtype=np.float16, shape=ds.shape)
        x_patch_mean = np.lib.format.open_memmap(tmp["patch_mean"], mode="w+", dtype=np.float16, shape=(len(ids), 768))
        x_token_mean = np.lib.format.open_memmap(tmp["token_mean"], mode="w+", dtype=np.float16, shape=(len(ids), 768))
        chunk = 128
        for start in range(0, len(ids), chunk):
            stop = min(start + chunk, len(ids))
            arr = np.asarray(ds[start:stop], dtype=np.float32)
            mean = arr.mean(axis=1).astype(np.float16)
            x_tokens[start:stop] = arr.astype(np.float16)
            x_patch_mean[start:stop] = mean
            x_token_mean[start:stop] = mean
            if stop <= 384 or stop % 1024 == 0 or stop == len(ids):
                print(f"[cache] {subject} {stop}/{len(ids)}", flush=True)
        x_tokens.flush()
        x_patch_mean.flush()
        x_token_mean.flush()
        del x_tokens, x_patch_mean, x_token_mean

    np.save(tmp["ids"], ids)
    for key in ("ids", "tokens", "patch_mean", "token_mean"):
        tmp[key].replace(paths[key])
    paths["metadata"].write_text(
        json.dumps(
            {
                "subject": subject,
                "source_hdf5": str(src),
                "n": len(ids),
                "token_shape": [256, 768],
                "dtype": "float16",
                "model": "siglip2-base-patch16-naflex",
                "embedding_label": embedding_label,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[cache] saved {embedding_label}_siglip2/{subject}: {paths['root']}",
        flush=True,
    )
    return paths


@dataclass
class FeatureStore:
    subject: str
    ids: np.ndarray
    x: np.ndarray
    index: dict[int, int]


def load_feature_store(args: argparse.Namespace, subject: str) -> FeatureStore:
    paths = ensure_hdf5_cache(
        hdf5_root=args.hdf5_root,
        hdf5_template=args.hdf5_template,
        cache_root=args.cache_root,
        subject=subject,
        embedding_label=args.embedding_label,
        force=args.force_cache,
    )
    ids = np.load(paths["ids"])
    x_path = paths["tokens"] if args.input_mode == "tokens256" else paths[args.one_token_source]
    x = np.load(x_path, mmap_mode="r")
    return FeatureStore(
        subject=subject,
        ids=ids,
        x=x,
        index={int(v): i for i, v in enumerate(ids.tolist())},
    )


class CocoFeatureDataset(Dataset):
    def __init__(
        self,
        stores: dict[str, FeatureStore],
        examples: list[tuple[str, int, int]],
        label_y: np.ndarray,
        label_index: dict[int, int],
    ):
        self.stores = stores
        self.examples = examples
        self.label_y = label_y
        self.label_index = label_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject, feature_idx, nsd_id = self.examples[idx]
        x_np = self.stores[subject].x[feature_idx]
        y_idx = self.label_index[int(nsd_id)]
        return (
            torch.from_numpy(np.array(x_np, dtype=np.float16, copy=True)),
            torch.from_numpy(np.array(self.label_y[y_idx], dtype=np.float32, copy=True)),
            torch.tensor(int(nsd_id), dtype=torch.long),
        )


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        dim: int,
        token_hidden: int,
        channel_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = MLPBlock(n_tokens, token_hidden, dropout)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = MLPBlock(dim, channel_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


def make_vector_mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class PatchTokenMixerClassifier(nn.Module):
    def __init__(
        self,
        n_patch_tokens: int = 256,
        in_dim: int = 768,
        hidden_dim: int = 768,
        depth: int = 4,
        token_hidden: int = 128,
        channel_hidden: int = 2048,
        classifier_hidden: int = 1024,
        classifier_depth: int = 2,
        dropout: float = 0.1,
        out_dim: int = 80,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    n_tokens=n_patch_tokens + 1,
                    dim=hidden_dim,
                    token_hidden=token_hidden,
                    channel_hidden=channel_hidden,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = make_vector_mlp(hidden_dim * 2, out_dim, classifier_hidden, classifier_depth, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.blocks(h)
        h = self.norm(h)
        pooled = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        return self.head(pooled)


class OneTokenClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.1,
        out_dim: int = 80,
    ):
        super().__init__()
        self.net = make_vector_mlp(in_dim, out_dim, hidden_dim, depth, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_examples(args: argparse.Namespace):
    shared, subject_ids = load_subject_membership(args.stim_info)
    label_ids = np.load(args.label_dir / "ids.npy")
    label_y = np.load(args.label_dir / "y_coco80.npy", mmap_mode="r")
    label_index = {int(nsd_id): idx for idx, nsd_id in enumerate(label_ids.tolist())}

    train_subjects = [args.test_subject] if args.regime == "within" else [s for s in SUBJECTS if s != args.test_subject]
    needed_subjects = sorted({*train_subjects, args.test_subject})
    stores = {subject: load_feature_store(args, subject) for subject in needed_subjects}

    def examples_for(subject: str, pool: set[int]) -> list[tuple[str, int, int]]:
        store = stores[subject]
        valid_ids = sorted(nsd_id for nsd_id in pool if nsd_id in store.index and nsd_id in label_index)
        return [(subject, store.index[int(nsd_id)], int(nsd_id)) for nsd_id in valid_ids]

    train_all: list[tuple[str, int, int]] = []
    for subject in train_subjects:
        pool = set(subject_ids[subject])
        pool.difference_update(shared)
        train_all.extend(examples_for(subject, pool))
    test_pool = set(subject_ids[args.test_subject]).intersection(shared)
    test_examples = examples_for(args.test_subject, test_pool)
    if len(train_all) < 2:
        raise ValueError(f"Insufficient train examples for {args.regime}/{args.test_subject}: {len(train_all)}")
    if not test_examples:
        raise ValueError(f"No test examples for {args.regime}/{args.test_subject}")

    rng = np.random.default_rng(args.seed)
    perm = np.arange(len(train_all))
    rng.shuffle(perm)
    n_val = max(1, round(len(perm) * args.val_frac)) if len(perm) > 1 else 0
    n_val = min(n_val, max(0, len(perm) - 1))
    val_examples = [train_all[int(i)] for i in perm[:n_val]]
    train_examples = [train_all[int(i)] for i in perm[n_val:]]
    return (
        stores,
        train_examples,
        val_examples,
        test_examples,
        label_y,
        label_index,
        train_subjects,
    )


def result_tag(
    input_mode: str,
    regime: str,
    test_subject: str,
    seed: int,
    one_token_source: str = "patch_mean",  # noqa: S107 - model token mode
    embedding_label: str = "nsd_roi",
) -> str:
    mode = f"one_token_{one_token_source}" if input_mode == "one_token" else input_mode
    return f"{safe_slug(embedding_label)}_siglip2_coco80_{mode}_{regime}_{test_subject}_seed{seed}"


def main() -> None:
    args = parse_args()
    if args.cache_only:
        ensure_hdf5_cache(
            hdf5_root=args.hdf5_root,
            hdf5_template=args.hdf5_template,
            cache_root=args.cache_root,
            subject=args.test_subject,
            embedding_label=args.embedding_label,
            force=args.force_cache,
        )
        return

    set_seed(args.seed)
    (
        stores,
        train_examples,
        val_examples,
        test_examples,
        label_y,
        label_index,
        train_subjects,
    ) = build_examples(args)
    train_ds = CocoFeatureDataset(stores, train_examples, label_y, label_index)
    val_ds = CocoFeatureDataset(stores, val_examples, label_y, label_index)
    test_ds = CocoFeatureDataset(stores, test_examples, label_y, label_index)
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    if args.input_mode == "tokens256":
        model = PatchTokenMixerClassifier(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            token_hidden=args.token_hidden,
            channel_hidden=args.channel_hidden,
            classifier_hidden=args.classifier_hidden,
            classifier_depth=args.classifier_depth,
            dropout=args.dropout,
            out_dim=80,
        ).to(device)
    else:
        model = OneTokenClassifier(
            hidden_dim=args.classifier_hidden,
            depth=args.classifier_depth + 1,
            dropout=args.dropout,
            out_dim=80,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_label_indices = [label_index[int(nsd_id)] for _, _, nsd_id in train_examples]
    train_y = np.asarray(label_y[train_label_indices], dtype=np.float32)
    pos = torch.from_numpy(train_y.sum(axis=0)).float()
    neg = float(len(train_y)) - pos
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), min=1.0, max=50.0).to(device)

    tag = result_tag(
        args.input_mode,
        args.regime,
        args.test_subject,
        args.seed,
        args.one_token_source,
        args.embedding_label,
    )
    out = args.out or (RESULTS_DIR / f"{tag}.json")
    ckpt = args.ckpt or (CKPT_DIR / f"{tag}.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] {args.embedding_label}_siglip2 input_mode={args.input_mode} one_token_source={args.one_token_source} "
        f"regime={args.regime} test_subject={args.test_subject} train_subjects={train_subjects} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} params={n_params:,} device={device} amp={amp}",
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
            f"val_micro_f1={val_metrics['micro_f1']:.4f} val_weighted_f1={val_metrics['weighted_f1']:.4f}",
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
        "task": f"{args.embedding_label} SigLIP2 patch tokens -> COCO80 direct MLP classification (subject split)",
        "model": "siglip2-base-patch16-naflex",
        "embedding_label": args.embedding_label,
        "input_mode": args.input_mode,
        "one_token_source": args.one_token_source,
        "regime": args.regime,
        "test_subject": args.test_subject,
        "train_subjects": train_subjects,
        "config": public_config(args),
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
