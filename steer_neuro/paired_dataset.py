"""Subject-paired (Champollion crop, ALMA crop) dataset for region S.C.-sylv.,
ALMA modality WholeBrain (see plan). Side (left/right) is selected via
STEER_NEURO_SIDE (default "right", so the original right-side pass keeps
working unchanged). Cohort (ukb/hcp/abcd) is selected via STEER_NEURO_COHORT
(default "ukb", same reasoning).

UKB and HCP subject universes both use bare IDs (`sub-XXXXXXX` for UKB,
plain numeric like `100206` for HCP) shared identically by Champollion's own
Subject column and ALMA's crop filename prefix, so pairing is a plain set
intersection for both -- no ID-cleaning transform needed. ABCD is the
exception (confirmed by comparing the two sides' real files, 2026-09-08):
Champollion's ABCD Subject column is the full `sub-NDARINVxxxxxxx`, but
ALMA's ABCD filenames drop the `NDARINV` and add a session suffix instead
(`sub-{xxxxxxx}_ses-00A_...`, multiple sessions per subject) -- ABCD's
alma_filename()/paired_subjects() below do that reconstruction + session
filtering explicitly (matching DeepLearning_Tracto/ALMA_ABCD_Prematurity.py's
own ID_clean convention and DEFAULT_SESSION="ses-00A").

Champollion side: the combined `Rskeleton.npy` array (+ `Rskeleton_subject.csv`
for row order), NOT individual per-subject files. Confirmed via
2025_Babdelghani_morphometric_y-aware/contrastive/data/{create_datasets.py,utils.py}
-- that project's own working Jean Zay pipeline for Champollion-style crops
reads skeleton data exclusively via `numpy_all` (`extract_data()` ->
`read_numpy_data_and_subject_csv(npy_file_path, subjects_all_csv)`); `crop_dir`
is only load-bearing for optional foldlabel/distbottom/extremity auxiliary
features, never for the plain skeleton crops this project needs. This also
matches what's actually on Jean Zay for S.C.-sylv./mask/ (only Rskeleton.npy +
Rskeleton_subject.csv, no Rcrops/) -- so this is the one loading path that
works in both places, not a Jean-Zay-specific special case.
`ChampoNpyDataset` below applies the exact same `TransCutRotMinMax`
augmentation pipeline (MinMax + Rotate3D + RandomCropAug + TranslateTensor)
that `preprocess.py::SkeletonDataset` uses, with matching default parameters,
so the two loading paths (this one and ALMA's file-based one) are
behaviorally equivalent.

ALMA side: still individual per-subject nifti files via
`preprocess.py::SkeletonDataset` (reused directly — pure numpy/nibabel/scipy/
torchvision, no lightning tax) — those crops were rsynced as individual files
to Jean Zay, same convention as local.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Overridable so this file (and everything it imports from DeepLearning_Tracto)
# can be pointed at Jean Zay's own lustre paths without code changes -- none of
# these neurospin paths are reachable from Jean Zay compute nodes.
_TRACTO_DIR = os.environ.get(
    "STEER_NEURO_TRACTO_DIR",
    os.path.join(os.path.dirname(__file__), "..", "DeepLearning_Tracto"),
)
sys.path.insert(0, _TRACTO_DIR)
from preprocess import SkeletonDataset, MinMax, Rotate3D, RandomCropAug, TranslateTensor  # noqa: E402
from torchvision import transforms  # noqa: E402

SIDE = os.environ.get("STEER_NEURO_SIDE", "right")
assert SIDE in ("left", "right"), f"STEER_NEURO_SIDE must be 'left' or 'right', got {SIDE!r}"
_PREFIX = "L" if SIDE == "left" else "R"  # Champollion's own crop-array naming convention

COHORT = os.environ.get("STEER_NEURO_COHORT", "ukb")
assert COHORT in ("ukb", "hcp", "abcd"), f"STEER_NEURO_COHORT must be ukb/hcp/abcd, got {COHORT!r}"
ABCD_SESSION = os.environ.get("STEER_NEURO_ABCD_SESSION", "ses-00A")  # matches ALMA_ABCD_Prematurity.py's DEFAULT_SESSION

# Dataset roots are site-specific (and the cohorts themselves are access
# controlled), so they come from the environment:
#   STEER_NEURO_CHAMPO_DATASET_DIR  skeleton crops for the chosen cohort
#   STEER_NEURO_ALMA_DATASET_DIR    ALMA crops for the chosen cohort
_CHAMPO_DATASET_DIR = os.environ.get("STEER_NEURO_CHAMPO_DATASET_DIR", "")
_ALMA_DATASET_DIR = os.environ.get("STEER_NEURO_ALMA_DATASET_DIR", "")

CHAMPO_SUBJECT_CSV = os.environ.get(
    "STEER_NEURO_CHAMPO_SUBJECT_CSV",
    f"{_CHAMPO_DATASET_DIR}/crops/2mm/S.C.-sylv./mask/{_PREFIX}skeleton_subject.csv",
)
CHAMPO_NPY = os.environ.get(
    "STEER_NEURO_CHAMPO_NPY",
    os.path.join(os.path.dirname(CHAMPO_SUBJECT_CSV), f"{_PREFIX}skeleton.npy"),
)
ALMA_CROP_DIR = os.environ.get(
    "STEER_NEURO_ALMA_CROP_DIR",
    f"{_ALMA_DATASET_DIR}/S.C.-sylv._{SIDE}/WholeBrain",
)
ALMA_REGION = f"S.C.-sylv._{SIDE}"
ALMA_MODALITY = "WholeBrain"
ALMA_MINL, ALMA_MAXL, ALMA_REFERENTIAL = 0, 250, "icbm09c"

_PREPROC_CFG = SimpleNamespace(trainer=SimpleNamespace(preproc="TransCutRotMinMax"))


def alma_filename(subject_id: str) -> str:
    if COHORT == "abcd":
        # champo IDs are "sub-NDARINVxxxxxxx"; ALMA's ABCD files drop NDARINV
        # and add a session suffix instead: "sub-xxxxxxx_ses-00A_...".
        short = subject_id.replace("sub-NDARINV", "sub-")
        return os.path.join(
            ALMA_CROP_DIR,
            f"{short}_{ABCD_SESSION}_{ALMA_REGION}_{ALMA_MODALITY}_{ALMA_MINL}_{ALMA_MAXL}_{ALMA_REFERENTIAL}_crop.nii.gz",
        )
    return os.path.join(
        ALMA_CROP_DIR,
        f"{subject_id}_{ALMA_REGION}_{ALMA_MODALITY}_{ALMA_MINL}_{ALMA_MAXL}_{ALMA_REFERENTIAL}_crop.nii.gz",
    )


def champo_subjects() -> list:
    """Full ordered Champollion subject list (== {L,R}skeleton.npy row order)."""
    return pd.read_csv(CHAMPO_SUBJECT_CSV)["Subject"].astype(str).tolist()


_PAIRED_SUBJECTS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "paired_subjects_cache")


def paired_subjects(limit: int = None):
    """Intersection of Champollion's and ALMA's subject universes for this
    region, filtered to subjects whose ALMA crop file actually exists (the
    Champollion side is an in-memory row lookup, no per-file existence check
    needed there). Returns a sorted list of subject IDs (cohort-native format:
    `sub-XXXXXXX` for ukb, plain numeric for hcp, `sub-NDARINVxxxxxxx` for abcd).

    Cached to disk per (COHORT, SIDE) -- the un-limited full list, keyed
    before `limit` is applied, so a --limit smoke test and a full run share
    the same cache entry. Confirmed 2026-09-10: computing this fresh does an
    os.listdir() over the whole ALMA_CROP_DIR PLUS one os.path.exists() per
    subject in the intersection (up to ~37k individual filesystem round-trips
    on ukb/left) -- easily minutes on a shared/networked filesystem, paid
    again on every single script invocation before training/extraction even
    starts. The crop directories are static preprocessing output that doesn't
    change during normal work here, so a forever-cache (no staleness check)
    is the right tradeoff; delete the cache file by hand if ALMA_CROP_DIR
    ever gets new subjects added to it."""
    os.makedirs(_PAIRED_SUBJECTS_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_PAIRED_SUBJECTS_CACHE_DIR, f"{COHORT}_{SIDE}.txt")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            both = [line.strip() for line in f if line.strip()]
    else:
        champo_subs = set(champo_subjects())
        if COHORT == "abcd":
            # only this session's files, then map "sub-xxxxxxx" back to champo's
            # "sub-NDARINVxxxxxxx" convention (see alma_filename's docstring note)
            alma_subs = set()
            for f in os.listdir(ALMA_CROP_DIR):
                if not f.endswith("_crop.nii.gz") or f"_{ABCD_SESSION}_" not in f:
                    continue
                short = f.split(f"_{ABCD_SESSION}_")[0]
                alma_subs.add(short.replace("sub-", "sub-NDARINV"))
        else:
            alma_subs = {
                f.split("_")[0] for f in os.listdir(ALMA_CROP_DIR) if f.endswith("_crop.nii.gz")
            }
        both = sorted(champo_subs & alma_subs)
        both = [s for s in both if os.path.exists(alma_filename(s))]
        with open(cache_path, "w") as f:
            f.write("\n".join(both) + ("\n" if both else ""))
    if limit is not None:
        both = both[:limit]
    return both


class ChampoNpyDataset(Dataset):
    """Champollion side, read from the combined {L,R}skeleton.npy array
    (shape (N, H, W, D, 1), int16 -- (42,34,49) right / (38,36,49) left)
    rather than individual nifti files. mmap'd, so this doesn't load the
    whole array into RAM."""

    def __init__(self, subjects):
        self.subjects = subjects
        self.arr = np.load(CHAMPO_NPY, mmap_mode="r")
        self._row_of = {s: i for i, s in enumerate(champo_subjects())}
        aug = [MinMax(), Rotate3D(), RandomCropAug(), TranslateTensor(max_shift=2)]
        self.transform = transforms.Compose(aug)
        self.transform_v2 = transforms.Compose(aug)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        row = self._row_of[sub]
        # (42, 34, 49, 1) -> (1, 42, 34, 49), matching SkeletonDataset's
        # np.expand_dims(img.get_fdata(), axis=0) channel-first convention
        raw = np.asarray(self.arr[row]).transpose(3, 0, 1, 2).astype(np.float32)
        return self.transform(raw), self.transform_v2(raw), sub


class PairedSkeletonDataset(Dataset):
    """Returns (champo_a, champo_b, alma_a, alma_b, subject_id) for each
    subject present in both modalities."""

    def __init__(self, subjects=None, limit: int = None):
        self.subjects = subjects if subjects is not None else paired_subjects(limit=limit)
        alma_paths = [alma_filename(s) for s in self.subjects]
        self.champo_ds = ChampoNpyDataset(self.subjects)
        self.alma_ds = SkeletonDataset(
            config=_PREPROC_CFG, file_paths=alma_paths, subject_ids=self.subjects
        )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        champo_a, champo_b, sub_c = self.champo_ds[idx]
        alma_a, alma_b, sub_a = self.alma_ds[idx]
        assert sub_c == sub_a, (sub_c, sub_a)
        return (
            torch.as_tensor(champo_a), torch.as_tensor(champo_b),
            torch.as_tensor(alma_a), torch.as_tensor(alma_b),
            sub_c,
        )


if __name__ == "__main__":
    subs = paired_subjects(limit=10)
    print(f"paired subjects (first 10 of full intersection): {subs}")
    ds = PairedSkeletonDataset(subjects=subs)
    print(f"dataset size: {len(ds)}")
    ca, cb, aa, ab, sub = ds[0]
    print(f"subject={sub}  champo view shapes={tuple(ca.shape)}/{tuple(cb.shape)}  "
          f"alma view shapes={tuple(aa.shape)}/{tuple(ab.shape)}")
