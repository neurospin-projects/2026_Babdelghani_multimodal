"""
MOSEI-MULTITASK — a SEPARATE dataset from `mosei`, with 7 labels instead of 1.

Deliberately self-contained: it does NOT touch CoMM's Affect loader, the `mosei`
catalog entry, or any existing checkpoint. `mosei` keeps behaving exactly as
before; `mosei_multitask` is a new dataset name with its own pickle, its own
loaders and its own probe tasks.

Data: built by pareto_ssl/multibench/build_mosei_multitask.py, which joins the
"All Labels" group of mosei.hdf5 onto mosei_senti_data.pkl by segment id. Same
features, labels of shape (N, 7):

    [sentiment, happy, sad, anger, surprise, disgust, fear]

Positive rates over the 23 248 segments: sentiment 49%, happy 54%, sad 26%,
anger 22%, disgust 18%, surprise 10%, fear 8%. USABLE_TASKS keeps the five in the
15-85% band; surprise/fear are too imbalanced to probe meaningfully.

Why it matters: this is the first REAL (non-synthetic) dataset in the benchmark
where several genuinely different labels share the same two modalities — i.e.
where "each task selects a different λ" is testable rather than structurally
impossible (single-label MOSEI has one optimum by construction).

Preprocessing replicates CoMM's Affect exactly (drop text-empty rows, leading-zero
trim, optional z-norm, per-batch padding), so numbers stay comparable to `mosei`.
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LABEL_NAMES = ["sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear"]
# tasks whose positive rate sits in a probe-able band (see module docstring)
USABLE_TASKS = ["sentiment", "happy", "sad", "anger", "disgust"]
DATASET_NAME = "mosei_multitask"
FEAT_DIM = {"vision": 35, "text": 300}
SPLIT_ALIAS = {"valid": "valid", "val": "valid", "train": "train", "test": "test"}


def is_multitask_dataset(dataset: str) -> bool:
    return dataset == DATASET_NAME


def task_index(task: str) -> int:
    return LABEL_NAMES.index(task)


def _drop_textless(d):
    """Same rule as CoMM's drop_entry: remove rows whose text is all zeros."""
    drop = [i for i, t in enumerate(d["text"]) if np.asarray(t).sum() == 0]
    if not drop:
        return d
    return {k: (np.delete(v, drop, 0) if hasattr(v, "__len__") and len(v) == len(d["text"]) else v)
            for k, v in d.items()}


class MoseiMultitask(Dataset):
    """Returns (X=[vision, text], y) with y a length-7 float vector.

    task=None            -> y is the raw 7-vector
    task="<name>"        -> y is a scalar 0/1 for that label (present = value > 0)
    """
    def __init__(self, data_path, split="train", modalities=("vision", "text"),
                 task=None, align=True, z_norm=False, ssl=False, augment=None):
        with open(data_path, "rb") as f:
            raw = pickle.load(f)
        key = SPLIT_ALIAS.get(split, split)
        if key not in raw:
            raise KeyError(f"split '{split}' not in {list(raw.keys())}")
        self.d = _drop_textless(dict(raw[key]))
        self.modalities = tuple(modalities)
        self.task = task
        self.align = align
        self.z_norm = z_norm
        self.ssl = ssl
        self.augment = augment
        y = np.asarray(self.d["labels"], dtype=np.float32)
        self.labels = y.reshape(len(y), -1)[:, :7]

    def __len__(self):
        return len(self.labels)

    def _mods(self, i):
        out = []
        for m in self.modalities:
            x = np.asarray(self.d[m][i])
            nz = x.nonzero()[0]
            if self.align:
                t = np.asarray(self.d["text"][i]).nonzero()[0]
                s = t[0] if len(t) else 0
            else:
                s = nz[0] if len(nz) else 0
            x = x[s:].astype(np.float32)
            if self.z_norm:
                x = np.nan_to_num((x - x.mean(0, keepdims=True)) / np.std(x, 0, keepdims=True))
            out.append(x)
        return out

    def _y(self, i):
        v = self.labels[i]
        if self.task is None:
            return v
        return np.float32(1.0 if v[task_index(self.task)] > 0 else 0.0)

    def __getitem__(self, i):
        X = self._mods(i)
        if self.ssl:
            a1 = self.augment(X) if self.augment else X
            a2 = self.augment(X) if self.augment else X
            return a1, a2
        return X, self._y(i)


DATASET_CLASS = MoseiMultitask


def _pad(seqs, max_len=None):
    """Pad a list of (T, p) arrays to (B, T*, p) — same convention as CoMM."""
    ts = [torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in seqs]
    T = max_len or max(t.shape[0] for t in ts)
    out = torch.zeros(len(ts), T, ts[0].shape[-1])
    for i, t in enumerate(ts):
        n = min(t.shape[0], T)
        out[i, :n] = t[:n]
    return out


def collate_probe(batch):
    n_mod = len(batch[0][0])
    X = [_pad([b[0][m] for b in batch]) for m in range(n_mod)]
    ys = [b[1] for b in batch]
    y = (torch.as_tensor(np.stack(ys)) if np.asarray(ys[0]).ndim else
         torch.as_tensor(np.asarray(ys, dtype=np.float32)))
    return X, y


def collate_ssl(batch):
    n_mod = len(batch[0][0])
    a1 = [_pad([b[0][m] for b in batch]) for m in range(n_mod)]
    a2 = [_pad([b[1][m] for b in batch]) for m in range(n_mod)]
    return a1, a2


def ssl_loader(data_path, batch_size, num_workers=4, augment=None, **kw):
    ds = MoseiMultitask(data_path, "train", ssl=True, augment=augment, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True, collate_fn=collate_ssl)


def probe_loader(data_path, split, batch_size, task=None, num_workers=4, **kw):
    ds = MoseiMultitask(data_path, split, task=task, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate_probe)


def summary(data_path):
    """Print split sizes and per-task positive rates — quick sanity check."""
    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    print(f"{os.path.basename(data_path)}")
    for split in raw:
        d = _drop_textless(dict(raw[split]))
        y = np.asarray(d["labels"], dtype=np.float32).reshape(len(d["labels"]), -1)[:, :7]
        rates = "  ".join(f"{n} {100*(y[:, i] > 0).mean():.0f}%" for i, n in enumerate(LABEL_NAMES))
        print(f"  {split:6s} n={len(y):6d}  {rates}")
    print(f"  usable tasks: {', '.join(USABLE_TASKS)}")


if __name__ == "__main__":
    import sys
    summary(sys.argv[1])
