from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURE_BASE = Path("outputs/features")
DEFAULT_PPMI_FEATURE_ROOT = DEFAULT_FEATURE_BASE / "ppmi" / "siglip2" / "naflex"
DEFAULT_ADNI_FEATURE_ROOT = DEFAULT_FEATURE_BASE / "adni" / "siglip2" / "naflex"
DEFAULT_HCP_FEATURE_ROOT = DEFAULT_FEATURE_BASE / "hcp" / "siglip2" / "naflex"
DEFAULT_HCP_ROI_ROOT = Path("data/splits/hcp/Schaefer2018_100")
DEFAULT_HCP_LABELS = Path("data/labels/hcp_labels.csv")
DEFAULT_PPMI_ROI_ROOT = Path("data/splits/ppmi/100ROI")
DEFAULT_PPMI_LABELS = Path("data/labels/ppmi_labels.csv")
DEFAULT_ADNI_SPLITS = Path("data/splits/adni")
DEFAULT_OUT_DIR = Path("outputs/resting_mlp")


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    subjects: list[str]
    feature_dirs: list[Path]


@dataclass
class TaskData:
    name: str
    display_name: str
    n_classes: int
    splits: dict[str, SplitData]


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, dim: int, token_hidden_dim: int, channel_hidden_dim: int, dropout: float):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, n_tokens),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class SequenceMLPMixer(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        dim: int,
        n_classes: int,
        depth: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MixerBlock(n_tokens, dim, token_hidden_dim, channel_hidden_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train downstream MLP probes on resting-state SigLIP2 NaFlex flatmap features.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--ppmi-feature-root", type=Path, default=DEFAULT_PPMI_FEATURE_ROOT)
    p.add_argument("--adni-feature-root", type=Path, default=DEFAULT_ADNI_FEATURE_ROOT)
    p.add_argument("--hcp-feature-root", type=Path, default=DEFAULT_HCP_FEATURE_ROOT)
    p.add_argument("--hcp-roi-root", type=Path, default=DEFAULT_HCP_ROI_ROOT)
    p.add_argument("--hcp-labels", type=Path, default=DEFAULT_HCP_LABELS)
    p.add_argument("--ppmi-roi-root", type=Path, default=DEFAULT_PPMI_ROI_ROOT)
    p.add_argument("--ppmi-labels", type=Path, default=DEFAULT_PPMI_LABELS)
    p.add_argument("--adni-splits", type=Path, default=DEFAULT_ADNI_SPLITS)
    p.add_argument("--tasks", default="hcp,ppmi,adni_mci,adni_ad")
    p.add_argument(
        "--feature-mode",
        choices=(
            "cls_flat",
            "cls_concat",
            "cls_seq",
            "cls_mean_token",
            "patch_mean_flat",
            "patch_mean_seq",
            "patch_mean_token",
            "cls_patch_flat",
        ),
        default="cls_flat",
    )
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--model-type", choices=("mlp", "mixer"), default="mlp")
    p.add_argument("--token-hidden-dim", type=int, default=64)
    p.add_argument("--channel-hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--best-metric", choices=("weighted_f1", "balanced_acc", "acc"), default="weighted_f1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-cache", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_from_bids_name(name: str) -> str:
    match = re.search(r"sub-([^_]+)", name)
    if not match:
        raise ValueError(f"Cannot parse subject from {name}")
    return match.group(1)


def subject_from_hcp_name(name: str) -> str:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", name)
    if not match:
        raise ValueError(f"Cannot parse HCP subject from {name}")
    return match.group(1)


def build_feature_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.glob("*.npz")):
        sid = subject_from_bids_name(path.stem)
        index.setdefault(sid, path)
    return index


def build_hcp_feature_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.glob("*.npz")):
        sid = subject_from_hcp_name(path.stem)
        index.setdefault(sid, path)
    return index


def feature_cache_path(out_dir: Path, dataset: str, mode: str) -> Path:
    return out_dir / "feature_cache" / f"{dataset}_{mode}.npy"


def _load_siglip_npz(feature_path: Path) -> dict[str, np.ndarray]:
    with np.load(feature_path) as data:
        out: dict[str, np.ndarray] = {}
        for key in data.files:
            value = data[key]
            if not np.issubdtype(value.dtype, np.number):
                continue
            out[key] = value.astype(np.float32, copy=False)
        return out


def _patch_mean_arrays(data: dict[str, np.ndarray]) -> np.ndarray:
    if "patch_mean" in data:
        return data["patch_mean"]
    if "patch_tokens_mean" in data:
        return data["patch_tokens_mean"]
    raise KeyError("SigLIP2 feature npz does not contain patch_mean or patch_tokens_mean.")


def load_subject_feature(feature_path: Path, mode: str) -> np.ndarray:
    data = _load_siglip_npz(feature_path)
    if "cls" not in data:
        raise KeyError(f"{feature_path} does not contain cls features.")
    if mode in ("cls_flat", "cls_concat"):
        return data["cls"].reshape(-1).astype(np.float32)
    if mode == "cls_seq":
        return data["cls"].astype(np.float32)
    if mode == "cls_mean_token":
        return data["cls"].mean(axis=0).astype(np.float32)
    if mode == "patch_mean_flat":
        return _patch_mean_arrays(data).reshape(-1).astype(np.float32)
    if mode == "patch_mean_seq":
        return _patch_mean_arrays(data).astype(np.float32)
    if mode == "patch_mean_token":
        return _patch_mean_arrays(data).mean(axis=0).astype(np.float32)
    if mode == "cls_patch_flat":
        cls = data["cls"].reshape(-1)
        patch = _patch_mean_arrays(data).reshape(-1)
        return np.concatenate([cls, patch], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported feature mode: {mode}")


def cached_feature_matrix(
    out_dir: Path,
    dataset: str,
    mode: str,
    subjects: list[str],
    feature_dirs: list[Path],
    force_cache: bool,
) -> np.ndarray:
    cache_dir = out_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = feature_cache_path(out_dir, dataset, mode)
    meta_path = x_path.with_suffix(".json")
    desired = {
        "dataset": dataset,
        "feature_mode": mode,
        "subjects": subjects,
        "feature_dirs": [str(p) for p in feature_dirs],
    }
    if x_path.exists() and meta_path.exists() and not force_cache:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta == desired:
                print(f"[cache] load {x_path}", flush=True)
                return np.load(x_path)
        except Exception:
            pass
    print(f"[cache] build {dataset} {mode} n={len(subjects)}", flush=True)
    rows: list[np.ndarray] = []
    for i, feature_dir in enumerate(feature_dirs, start=1):
        rows.append(load_subject_feature(feature_dir, mode))
        if i <= 5 or i % 50 == 0 or i == len(feature_dirs):
            print(f"[cache] {dataset} {i}/{len(feature_dirs)}", flush=True)
    x = np.stack(rows, axis=0).astype(np.float32, copy=False)
    np.save(x_path, x)
    meta_path.write_text(json.dumps(desired, indent=2), encoding="utf-8")
    return x


def ppmi_split_subjects(roi_root: Path, split: str) -> list[str]:
    subjects: set[str] = set()
    for path in sorted((roi_root / split).glob("*.npy")):
        subjects.add(subject_from_bids_name(path.name))
    return sorted(subjects)


def ppmi_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("Subject")) or pd.isna(row.get("DX_GROUP")):
            continue
        sid = str(int(row["Subject"])) if isinstance(row["Subject"], (int, float, np.integer, np.floating)) else str(row["Subject"]).strip()
        labels[sid] = int(row["DX_GROUP"])
    return labels


def hcp_split_subjects(roi_root: Path, split: str) -> list[str]:
    subjects: set[str] = set()
    for path in sorted((roi_root / split).glob("*.npy")):
        subjects.add(subject_from_hcp_name(path.name))
    return sorted(subjects)


def hcp_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    labels: dict[str, int] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("Subject")) or pd.isna(row.get("Gender")):
            continue
        sid = subject_from_hcp_name(str(row["Subject"]))
        gender = int(row["Gender"])
        if gender in (0, 1):
            labels[sid] = gender
    return labels


def build_hcp_task(args: argparse.Namespace, feature_index: dict[str, Path]) -> TaskData:
    labels = hcp_labels(args.hcp_labels)
    splits: dict[str, SplitData] = {}
    all_subjects: list[str] = []
    all_dirs: list[Path] = []
    split_subject_lists: dict[str, list[str]] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        subjects = [sid for sid in hcp_split_subjects(args.hcp_roi_root, split) if sid in labels and sid in feature_index]
        split_subject_lists[split] = subjects
        split_labels[split] = np.asarray([labels[sid] for sid in subjects], dtype=np.int64)
        all_subjects.extend(subjects)
        all_dirs.extend([feature_index[sid] for sid in subjects])
    x_all = cached_feature_matrix(args.out_dir, "hcp", args.feature_mode, all_subjects, all_dirs, args.force_cache)
    offset = 0
    for split in ("train", "val", "test"):
        n = len(split_subject_lists[split])
        subjects = split_subject_lists[split]
        dirs = [feature_index[sid] for sid in subjects]
        splits[split] = SplitData(x=x_all[offset : offset + n], y=split_labels[split], subjects=subjects, feature_dirs=dirs)
        offset += n
    return TaskData("hcp", "HCP Sex Classif.", 2, splits)


def build_ppmi_task(args: argparse.Namespace, feature_index: dict[str, Path]) -> TaskData:
    labels = ppmi_labels(args.ppmi_labels)
    splits: dict[str, SplitData] = {}
    all_subjects: list[str] = []
    all_dirs: list[Path] = []
    split_subject_lists: dict[str, list[str]] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        subjects = [sid for sid in ppmi_split_subjects(args.ppmi_roi_root, split) if sid in labels and sid in feature_index]
        split_subject_lists[split] = subjects
        split_labels[split] = np.asarray([labels[sid] for sid in subjects], dtype=np.int64)
        all_subjects.extend(subjects)
        all_dirs.extend([feature_index[sid] for sid in subjects])
    x_all = cached_feature_matrix(args.out_dir, "ppmi", args.feature_mode, all_subjects, all_dirs, args.force_cache)
    offset = 0
    for split in ("train", "val", "test"):
        n = len(split_subject_lists[split])
        subjects = split_subject_lists[split]
        dirs = [feature_index[sid] for sid in subjects]
        splits[split] = SplitData(x=x_all[offset : offset + n], y=split_labels[split], subjects=subjects, feature_dirs=dirs)
        offset += n
    return TaskData("ppmi", "PPMI PD Diagnosis", 3, splits)


def load_adni_subjects(split_root: Path, group: str, split: str) -> list[str]:
    path = split_root / f"ADNI_atlas_{group}" / f"{split}.csv"
    df = pd.read_csv(path)
    return [str(v).strip().removeprefix("sub-") for v in df["Subject"].tolist()]


def build_adni_task(args: argparse.Namespace, feature_index: dict[str, Path], positive: str, name: str, display: str) -> TaskData:
    splits: dict[str, SplitData] = {}
    all_subjects: list[str] = []
    all_dirs: list[Path] = []
    split_subject_lists: dict[str, list[str]] = {}
    split_y: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        cn = [sid for sid in load_adni_subjects(args.adni_splits, "cn", split) if sid in feature_index]
        pos = [sid for sid in load_adni_subjects(args.adni_splits, positive, split) if sid in feature_index]
        subjects = cn + pos
        split_subject_lists[split] = subjects
        split_y[split] = np.asarray([0] * len(cn) + [1] * len(pos), dtype=np.int64)
        all_subjects.extend(subjects)
        all_dirs.extend([feature_index[sid] for sid in subjects])
    x_all = cached_feature_matrix(args.out_dir, name, args.feature_mode, all_subjects, all_dirs, args.force_cache)
    offset = 0
    for split in ("train", "val", "test"):
        n = len(split_subject_lists[split])
        subjects = split_subject_lists[split]
        dirs = [feature_index[sid] for sid in subjects]
        splits[split] = SplitData(x=x_all[offset : offset + n], y=split_y[split], subjects=subjects, feature_dirs=dirs)
        offset += n
    return TaskData(name, display, 2, splits)


def standardize_task(task: TaskData) -> TaskData:
    mean = task.splits["train"].x.mean(axis=0, keepdims=True)
    std = task.splits["train"].x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    splits = {}
    for split, data in task.splits.items():
        splits[split] = SplitData(
            x=((data.x - mean) / std).astype(np.float32),
            y=data.y,
            subjects=data.subjects,
            feature_dirs=data.feature_dirs,
        )
    return TaskData(task.name, task.display_name, task.n_classes, splits)


def class_weights(y: np.ndarray, n_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32, copy=False)), torch.from_numpy(y.astype(np.int64, copy=False)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = prob.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }
    try:
        if prob.shape[1] == 2:
            out["auc"] = float(roc_auc_score(y_true, prob[:, 1]))
        else:
            out["auc_ovr_weighted"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="weighted"))
    except Exception:
        pass
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        losses.append(float(F.cross_entropy(out, yb).detach().cpu()))
        ys.append(yb.detach().cpu().numpy())
        logits.append(out.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(logits, axis=0)


def train_one_seed(task: TaskData, seed: int, args: argparse.Namespace) -> dict:
    set_seed(seed)
    device = args.device
    train = task.splits["train"]
    val = task.splits["val"]
    test = task.splits["test"]
    train_loader = make_loader(train.x, train.y, args.batch_size, True)
    val_loader = make_loader(val.x, val.y, args.batch_size, False)
    test_loader = make_loader(test.x, test.y, args.batch_size, False)
    if args.model_type == "mixer":
        if train.x.ndim != 3:
            raise ValueError(f"model-type=mixer expects sequence features, got shape {train.x.shape}")
        model = SequenceMLPMixer(
            train.x.shape[1],
            train.x.shape[2],
            task.n_classes,
            args.depth,
            args.token_hidden_dim,
            args.channel_hidden_dim,
            args.dropout,
        ).to(device)
    else:
        if train.x.ndim != 2:
            raise ValueError(f"model-type=mlp expects flattened features, got shape {train.x.shape}")
        model = MLP(train.x.shape[1], task.n_classes, args.hidden_dim, args.depth, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    weights = class_weights(train.y, task.n_classes, device)
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_loss, y_val, val_logits = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_val, val_logits)
        score = float(val_metrics[args.best_metric])
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, **val_metrics})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_loss, y_train, train_logits = evaluate(model, train_loader, device)
    val_loss, y_val, val_logits = evaluate(model, val_loader, device)
    test_loss, y_test, test_logits = evaluate(model, test_loader, device)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_metrics": compute_metrics(y_train, train_logits),
        "val_metrics": compute_metrics(y_val, val_logits),
        "test_metrics": compute_metrics(y_test, test_logits),
        "history": history,
    }


def summarize_runs(runs: list[dict]) -> dict[str, dict[str, float]]:
    metrics = sorted({key for run in runs for key in run["test_metrics"]})
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        vals = np.asarray([run["test_metrics"][metric] for run in runs if metric in run["test_metrics"]], dtype=float)
        out[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals.tolist(),
        }
    return out


def save_selected_files(path: Path, tasks: list[TaskData]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "split", "subject", "label", "feature_dir"])
        for task in tasks:
            for split, data in task.splits.items():
                for sid, y, feature_dir in zip(data.subjects, data.y, data.feature_dirs):
                    writer.writerow([task.name, split, sid, int(y), str(feature_dir)])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    requested = {x.strip() for x in args.tasks.split(",") if x.strip()}

    tasks: list[TaskData] = []
    if "hcp" in requested:
        hcp_index = build_hcp_feature_index(args.hcp_feature_root)
        tasks.append(build_hcp_task(args, hcp_index))
    if "ppmi" in requested:
        ppmi_index = build_feature_index(args.ppmi_feature_root)
        tasks.append(build_ppmi_task(args, ppmi_index))
    if "adni_mci" in requested:
        adni_index = build_feature_index(args.adni_feature_root)
        tasks.append(build_adni_task(args, adni_index, "mci", "adni_mci", "ADNI (MCI) Diagnosis"))
    if "adni_ad" in requested:
        adni_index = build_feature_index(args.adni_feature_root)
        tasks.append(build_adni_task(args, adni_index, "ad", "adni_ad", "ADNI (AD) Diagnosis"))
    if not tasks:
        raise RuntimeError("No tasks requested.")

    save_selected_files(args.out_dir / f"selected_subject_files_{args.feature_mode}.csv", tasks)
    tasks = [standardize_task(task) for task in tasks]

    results: dict[str, dict] = {}
    rows: list[dict] = []
    for task in tasks:
        print(f"=== {task.name} {task.display_name}", flush=True)
        print(
            "split counts",
            {split: int(len(data.y)) for split, data in task.splits.items()},
            "class counts",
            {split: np.bincount(data.y, minlength=task.n_classes).astype(int).tolist() for split, data in task.splits.items()},
            "feature_dim",
            int(task.splits["train"].x.shape[1]),
            flush=True,
        )
        runs = []
        for seed in seeds:
            print(f"task={task.name} seed={seed}", flush=True)
            run = train_one_seed(task, seed, args)
            runs.append(run)
            tm = run["test_metrics"]
            print(
                f"seed={seed} epoch={run['best_epoch']} "
                f"acc={tm['acc']:.4f} wF1={tm['weighted_f1']:.4f} bal={tm['balanced_acc']:.4f}",
                flush=True,
            )
            rows.append({"task": task.name, "display_name": task.display_name, "seed": seed, "best_epoch": run["best_epoch"], **{f"test_{k}": v for k, v in tm.items()}})
        summary = summarize_runs(runs)
        results[task.name] = {
            "display_name": task.display_name,
            "n_classes": task.n_classes,
            "split_counts": {split: int(len(data.y)) for split, data in task.splits.items()},
            "class_counts": {split: np.bincount(data.y, minlength=task.n_classes).astype(int).tolist() for split, data in task.splits.items()},
            "feature_dim": int(task.splits["train"].x.shape[1]),
            "runs": runs,
            "summary": summary,
        }

    config = {
        "feature": "resting global_zscore_clip3 SigLIP2 NaFlex 40-frame npz features",
        "feature_mode": args.feature_mode,
        "ppmi_feature_root": str(args.ppmi_feature_root),
        "adni_feature_root": str(args.adni_feature_root),
        "classifier": {
            "type": "MLP",
            "model_type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "token_hidden_dim": args.token_hidden_dim,
            "channel_hidden_dim": args.channel_hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "best_metric": args.best_metric,
            "class_weighted_ce": True,
        },
        "seeds": seeds,
    }
    out = {"config": config, "results": results}
    (args.out_dir / f"resting_siglip2_naflex_mlp_{args.feature_mode}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out_dir / f"per_run_metrics_{args.feature_mode}.csv", index=False)

    summary_rows = []
    for name, item in results.items():
        summary = item["summary"]
        summary_rows.append(
            {
                "task": name,
                "display_name": item["display_name"],
                "acc_mean": summary["acc"]["mean"],
                "acc_std": summary["acc"]["std"],
                "weighted_f1_mean": summary["weighted_f1"]["mean"],
                "weighted_f1_std": summary["weighted_f1"]["std"],
                "balanced_acc_mean": summary["balanced_acc"]["mean"],
                "balanced_acc_std": summary["balanced_acc"]["std"],
                "macro_f1_mean": summary["macro_f1"]["mean"],
                "macro_f1_std": summary["macro_f1"]["std"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.out_dir / f"summary_metrics_{args.feature_mode}.csv", index=False)
    print("\nSummary percent (mean+-std):", flush=True)
    for row in summary_rows:
        print(
            f"{row['display_name']}: ACC {row['acc_mean'] * 100:.2f}+-{row['acc_std'] * 100:.2f}, "
            f"wF1 {row['weighted_f1_mean'] * 100:.2f}+-{row['weighted_f1_std'] * 100:.2f}",
            flush=True,
        )
    print(f"saved {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
