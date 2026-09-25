"""Trifeature corr=0 dataset — 2 modalities (M1, M2) for Pareto-SSL benchmark.

Images are resized to 224×224 to match AlexNet's expected input size.
Dataset scale matches CoMM trifeatures notebook: N_TRAIN=2400 / N_TEST=200.
Probing uses PAIRS (M1_i, M2_j) with matching shape class R_i == R_j,
following the same design as CoMM's BimodalTrifeatures.
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def _remap(stored: str, data_dir: str) -> str:
    """Remap an absolute path stored in metadata.json to the actual data_dir.
    Extracts the last two path components (subfolder/file.png) and joins with data_dir.
    Falls back to the stored path if it already exists (local machine).
    """
    if os.path.exists(stored):
        return stored
    p = Path(stored)
    return str(Path(data_dir) / p.parent.name / p.name)

# Tasks
TASKS = ["share", "unique1", "unique2"]  # = R, U1, U2 but pair-based

# which two modalities form the pair
# Every record renders THREE images of the same shape R, each carrying one extra
# factor: M1 deformation (U1), M2 texture (U2), M3 colour (U3). Experiments used the
# M1/M2 pair exclusively; "M3,M2" gives R = shape, unique1 = colour, unique2 = texture.
#
# Why it matters: U1 is a DEFORMATION OF THE SHAPE, so it is entangled with R by
# construction, and shape leaks into the M1-unique head (1.448 bits at that vertex vs
# 0.942 for texture). Colour is separable from shape, so the pair choice is a direct
# test of whether that leak comes from factor entanglement or from the shared base.
#
# Set through set_pair() (benchmark.py --modality-pair, recorded in train_meta.json) or
# the TRIFEATURE_PAIR env var. Default keeps every existing result reproducible.
_MOD_UNIQUE = {"M1": "U1", "M2": "U2", "M3": "U3"}
PAIR = ["M1", "M2"]


def set_pair(spec):
    """spec: "M3,M2" or ("M3", "M2"). Returns the normalised pair."""
    global PAIR
    parts = [x.strip().upper() for x in (spec.split(",") if isinstance(spec, str) else spec)]
    if len(parts) != 2 or any(x not in _MOD_UNIQUE for x in parts):
        raise ValueError(f"modality pair must be two of {sorted(_MOD_UNIQUE)}, got {spec!r}")
    if parts[0] == parts[1]:
        raise ValueError(f"modality pair must use two different modalities, got {spec!r}")
    PAIR = parts
    return list(PAIR)


def get_pair():
    """Current pair as a list — read this, not the PAIR name, since set_pair rebinds it."""
    return list(PAIR)


def pair_path_keys():
    return f"{PAIR[0]}_path", f"{PAIR[1]}_path"


def pair_unique_keys():
    """Label keys behind tasks unique1 / unique2 for the current pair."""
    return _MOD_UNIQUE[PAIR[0]], _MOD_UNIQUE[PAIR[1]]


if os.environ.get("TRIFEATURE_PAIR"):
    set_pair(os.environ["TRIFEATURE_PAIR"])

# Match CoMM trifeatures notebook: 2400 train / 200 test images
N_TRAIN = 2400
N_TEST  = 200

# Transforms
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

_EVAL = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORM,
])

# Standard SimCLR recipe. NOTE what it does to the unique objective: L_self is NT-Xent
# between two augmented views, so the unique head is trained to be INVARIANT to whatever
# the augmentation changes. ColorJitter(0.4) at p=0.8 plus RandomGrayscale(p=0.2) removes
# colour, so an M3 (colour-unique) head is explicitly optimised to discard the very factor
# it is later probed for -- measured: U1 0.782 bits with this pipeline, and the shape leak
# at the U1 vertex grows to 2.333 because shape is the only augmentation-invariant content
# left. Deformation survives crop/flip far better (U1 1.111, leak 1.448).
_AUG = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    _NORM,
])

# Colour-preserving variant: geometry only. For a modality whose UNIQUE factor is colour,
# this is the augmentation that lets the factor survive contrastive training at all.
_AUG_NOCOLOR = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _NORM,
])

# Per-modality augmentation, indexed like PAIR ("M1","M2","M3"). Default: the SimCLR
# recipe everywhere, which reproduces every existing result. set_augment() overrides it,
# e.g. set_augment("M3:nocolor") for the colour pair.
_AUG_BY_NAME = {"simclr": _AUG, "nocolor": _AUG_NOCOLOR}
AUG_PER_MOD = {"M1": "simclr", "M2": "simclr", "M3": "simclr"}


def set_augment(spec):
    """spec: "M3:nocolor" or "M1:simclr,M3:nocolor". Returns the resulting mapping."""
    if not spec:
        return dict(AUG_PER_MOD)
    for part in spec.split(","):
        mod, _, name = part.partition(":")
        mod, name = mod.strip().upper(), name.strip().lower()
        if mod not in AUG_PER_MOD:
            raise ValueError(f"unknown modality {mod!r}; expected one of {sorted(AUG_PER_MOD)}")
        if name not in _AUG_BY_NAME:
            raise ValueError(f"unknown augmentation {name!r}; expected one of {sorted(_AUG_BY_NAME)}")
        AUG_PER_MOD[mod] = name
    return dict(AUG_PER_MOD)


def get_augment():
    return dict(AUG_PER_MOD)


def _aug_for(mod):
    return _AUG_BY_NAME[AUG_PER_MOD[mod]]


class TrifeatureTrainDataset(Dataset):
    """
    Returns two independently augmented views for M1 and M2:
        [(M1_a, M1_b), (M2_a, M2_b)]

    All methods pick what they need:
        - within-M1 (SimCLR…)       : use views[0]
        - cross-modal (CLIP, GMC…)  : use views[0][0] and views[1][0]
        - FactorCL                  : uses all four views
    """
    def __init__(self, data_dir: str, n_train: int = N_TRAIN):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json")) as f:
            records = json.load(f)
        self.records = records[:n_train]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        views = []
        for mod, key in zip(PAIR, pair_path_keys()):
            img = Image.open(_remap(r[key], self.data_dir)).convert("RGB")
            aug = _aug_for(mod)          # per-modality: see AUG_PER_MOD
            views.append((aug(img), aug(img)))
        return views   # [(Ma_a, Ma_b), (Mb_a, Mb_b)]


class PairProbeDataset(Dataset):
    """
    Pair-based probe dataset following CoMM's BimodalTrifeatures design.

    Each item: ([M1_i, M2_j], label) where R_i == R_j but i != j.

    Tasks:
        share   → label = R_i  (shared shape class, 10 classes)
        unique1 → label = U1_i  (M1's unique deformation, 10 classes)
        unique2 → label = U2_j  (M2's unique texture, 10 classes)

    This pair-based design ensures:
    - Methods that capture shared info score high on 'share'
    - Methods that capture M1 unique info score high on 'unique1'
    - Methods that capture M2 unique info score high on 'unique2'
    """
    def __init__(self, data_dir: str, split: str = "test",
                 task: str = "share", max_pairs: int = 10_000, seed: int = 42):
        assert task in TASKS, f"task must be one of {TASKS}"
        assert split in {"train", "val", "test"}

        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json")) as f:
            records = json.load(f)
        # Match CoMM notebook scale: 2400 train / 200 val / 200 test
        splits = {
            "train": records[:N_TRAIN],
            "val":   records[N_TRAIN: N_TRAIN + N_TEST],
            "test":  records[N_TRAIN + N_TEST: N_TRAIN + 2 * N_TEST],
        }
        recs = splits[split]
        self.task = task

        # Group record indices by R value
        from collections import defaultdict
        r_groups = defaultdict(list)
        for idx, rec in enumerate(recs):
            r_groups[rec["R"]].append(idx)

        rng = np.random.default_rng(seed)

        # Build pairs: (M1 from record i, M2 from record j) with R_i == R_j, i != j
        pairs = []
        n_per_class = max(1, max_pairs // len(r_groups))
        for r_val, idxs in r_groups.items():
            idxs = np.array(idxs)
            if len(idxs) < 2:
                continue
            # Sample pairs within this R class
            n_sample = min(n_per_class, len(idxs) ** 2 - len(idxs))
            drawn = 0
            attempts = 0
            while drawn < n_sample and attempts < n_sample * 10:
                i, j = rng.choice(len(idxs), size=2, replace=False)
                pairs.append((idxs[i], idxs[j]))
                drawn += 1
                attempts += 1

        self.pairs = pairs
        self.records = recs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        ri, rj = self.records[i], self.records[j]

        k1, k2 = pair_path_keys()
        u1_key, u2_key = pair_unique_keys()
        m1 = _EVAL(Image.open(_remap(ri[k1], self.data_dir)).convert("RGB"))
        m2 = _EVAL(Image.open(_remap(rj[k2], self.data_dir)).convert("RGB"))

        if self.task == "share":
            label = ri["R"]          # == rj["R"] by construction
        elif self.task == "unique1":
            label = ri[u1_key]       # first modality's unique factor
        elif self.task == "unique2":
            label = rj[u2_key]       # second modality's unique factor
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return [m1, m2], label
