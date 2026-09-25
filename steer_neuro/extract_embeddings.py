"""Extract STEER-neuro embeddings z(lambda) for UKB subjects, either at the 4
named vertices or across the full 15-point simplex grid, for downstream
evaluation against ALMA's own precomputed embeddings (see eval_downstream.py)
and for lambda-selection on future downstream labels (see select_lambda.py).

The STEER strategy is to ship the *family* z(lambda), not a single embedding:
which lambda is best is downstream-task-dependent (see
select_lambda.py / project_steer_method_final), so --grid extracts all 15
grid points instead of just the 4 named vertices, giving any future
downstream analysis the full family to pick from without re-running
extraction.

Deterministic: MinMax-only preprocessing, no augmentation, single view --
unlike training (which needs two augmented views per sample), inference
should not vary run to run.

Usage:
  python3 extract_embeddings.py --ckpt <path to steer_neuro_final.pt> \
    --out_dir <dir> [--points R U_champo U_alma centroid] [--limit N]
  python3 extract_embeddings.py --ckpt <ckpt> --out_dir <dir> --grid  # all 15 points
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from model import SteerNeuroModel
from paired_dataset import alma_filename, champo_subjects, paired_subjects, CHAMPO_NPY
from simplex import dim_readout, dim_readout_pca, fit_block_pca, simplex_grid
from wrap_lora import SCOPES

_TRACTO_DIR = os.environ.get(
    "STEER_NEURO_TRACTO_DIR",
    os.path.join(os.path.dirname(__file__), "..", "DeepLearning_Tracto"),
)
sys.path.insert(0, _TRACTO_DIR)
from preprocess import SkeletonDataset, MinMax  # noqa: E402
from torchvision import transforms  # noqa: E402

_INFER_CFG = SimpleNamespace(trainer=SimpleNamespace(preproc="Inference"))

PREFERENCE_POINTS = {
    "R": (1.0, 0.0, 0.0),
    "U_champo": (0.0, 1.0, 0.0),
    "U_alma": (0.0, 0.0, 1.0),
    "centroid": (1 / 3, 1 / 3, 1 / 3),
}


class ChampoNpyInferenceDataset(Dataset):
    """Deterministic (MinMax only) Champollion loader -- see ChampoNpyDataset
    in paired_dataset.py for the augmented, training-time version."""

    def __init__(self, subjects):
        self.subjects = subjects
        self.arr = np.load(CHAMPO_NPY, mmap_mode="r")
        self._row_of = {s: i for i, s in enumerate(champo_subjects())}
        self.transform = transforms.Compose([MinMax()])

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        row = self._row_of[sub]
        raw = np.asarray(self.arr[row]).transpose(3, 0, 1, 2).astype(np.float32)
        return self.transform(raw), sub


class PairedInferenceDataset(Dataset):
    """Returns (champo, alma, subject_id) -- single deterministic view each,
    for embedding extraction (not training)."""

    def __init__(self, subjects):
        self.subjects = subjects
        self.champo_ds = ChampoNpyInferenceDataset(subjects)
        alma_paths = [alma_filename(s) for s in subjects]
        self.alma_ds = SkeletonDataset(config=_INFER_CFG, file_paths=alma_paths, subject_ids=subjects)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        c, sub_c = self.champo_ds[idx]
        a, sub_a = self.alma_ds[idx]
        assert sub_c == sub_a, (sub_c, sub_a)
        return torch.as_tensor(c), torch.as_tensor(a), sub_c


def _grid_name(lam):
    """Filename-safe tag for a raw grid point, e.g. (0.75, 0.25, 0.0) -> '0.75_0.25_0.00'."""
    return "_".join(f"{x:.2f}" for x in lam)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--points", nargs="+", default=list(PREFERENCE_POINTS.keys()),
                   choices=list(PREFERENCE_POINTS.keys()),
                   help="named preference points to extract (default: all 4). Ignored if --grid.")
    p.add_argument("--grid", action="store_true",
                   help="extract all 15 simplex-grid points instead of --points (the STEER "
                        "'ship the family' strategy -- see select_lambda.py for why)")
    p.add_argument("--limit", type=int, default=None, help="subject-count cap")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--total_dim", type=int, default=None, help="defaults to proj_dim (the 'dim' readout width, see the method description sec:readout)")
    p.add_argument("--readout", choices=["dim", "dim_pca"], default="dim",
                   help="'dim' (default): mixed-lambda points keep the first d_b raw "
                        "coordinates of each block "
                        "'dim_pca': keep the top-d_b PCA components of each block instead, "
                        "fit per lambda point on this extraction's own full population "
                        "(see simplex.dim_readout_pca). Only changes MIXED grid points -- "
                        "pure vertices get a block's full width either way and are "
                        "numerically identical between the two readouts.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(ckpt_path, device):
    """Reconstruct a SteerNeuroModel from a training checkpoint and load its
    weights. Returns (model, train_args) -- shared by extract_embeddings.py's
    main() and select_lambda.py so checkpoint-loading logic lives in one
    place."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    assert train_args["scope"] in SCOPES

    model = SteerNeuroModel(
        scope=train_args["scope"], rank=train_args["rank"],
        rank_champo=train_args.get("rank_champo"), rank_alma=train_args.get("rank_alma"),
        rank_ratio=train_args.get("rank_ratio"),
        rank_ratio_champo=train_args.get("rank_ratio_champo"),
        rank_ratio_alma=train_args.get("rank_ratio_alma"),
        alpha=train_args["alpha"], alpha_champo=train_args.get("alpha_champo"),
        alpha_alma=train_args.get("alpha_alma"), proj_dim=train_args["proj_dim"],
        head_hidden=train_args["head_hidden"], club_hidden=train_args["club_hidden"],
        beta_club=train_args["beta_club"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, train_args


def main():
    args = parse_args()
    model, train_args = load_model(args.ckpt, args.device)

    # the "dim" readout's width is D
    # (one block's own width), NOT 3*D (that's the "None"/capacity-control
    # readout's width, a different thing entirely). Fixed 2026-09-08 --
    # every embedding extracted before this fix used the wrong 3*D budget.
    total_dim = args.total_dim or train_args["proj_dim"]

    if args.grid:
        points_map = {_grid_name(lam): lam for lam in simplex_grid()}
        print(f"[points] --grid: extracting all {len(points_map)} simplex-grid points")
    else:
        points_map = {name: PREFERENCE_POINTS[name] for name in args.points}

    subjects = paired_subjects(limit=args.limit)
    print(f"[data] {len(subjects)} paired subjects", flush=True)
    ds = PairedInferenceDataset(subjects)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    accum = {name: [] for name in points_map}
    # dim_pca needs the FULL population's raw blocks before it can fit a PCA
    # basis and select top components -- unlike 'dim', which is a stateless
    # per-sample slice, so it can be applied streaming as each batch arrives.
    # Buffered here instead: still exactly one encoder forward per (batch,
    # lambda) either way, this only defers the cheap CPU readout-combination
    # step to after the full pass (see simplex.fit_block_pca/dim_readout_pca).
    raw_blocks = ({name: {"r": [], "uc": [], "ua": []} for name in points_map}
                  if args.readout == "dim_pca" else None)
    all_subs = []
    with torch.no_grad():
        for champo, alma, subs in tqdm(loader, desc="extracting", unit="batch"):
            champo, alma = champo.to(args.device), alma.to(args.device)
            all_subs.extend(subs)
            for name, lam in points_map.items():
                r, uc, ua = model.embed_blocks(champo, alma, *lam)
                if args.readout == "dim_pca":
                    raw_blocks[name]["r"].append(r.cpu())
                    raw_blocks[name]["uc"].append(uc.cpu())
                    raw_blocks[name]["ua"].append(ua.cpu())
                else:
                    z = dim_readout([r, uc, ua], list(lam), total_dim=total_dim)
                    accum[name].append(z.cpu().numpy())

    if args.readout == "dim_pca":
        print("[readout] dim_pca: fitting per-lambda PCA on this extraction's "
              "full population...", flush=True)
        for name, lam in points_map.items():
            r_full = torch.cat(raw_blocks[name]["r"], dim=0)
            uc_full = torch.cat(raw_blocks[name]["uc"], dim=0)
            ua_full = torch.cat(raw_blocks[name]["ua"], dim=0)
            pcas = fit_block_pca([r_full, uc_full, ua_full])
            z = dim_readout_pca([r_full, uc_full, ua_full], list(lam),
                                 total_dim=total_dim, pcas=pcas)
            accum[name].append(z.numpy())

    # ALMA's own residualized embeddings key on bare numeric UKB IDs (no "sub-"
    # prefix) -- match that convention so eval_downstream.py can join directly.
    bare_subs = [s.replace("sub-", "") for s in all_subs]
    for name in points_map:
        z = np.concatenate(accum[name], axis=0)
        df = pd.DataFrame(z, columns=[str(i) for i in range(z.shape[1])])
        df.insert(0, "subject", bare_subs)
        out_path = os.path.join(args.out_dir, f"steer_neuro_ukb_lam-{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[{name}] saved {out_path} ({len(df)} subjects, {z.shape[1]} dims)", flush=True)


if __name__ == "__main__":
    main()
