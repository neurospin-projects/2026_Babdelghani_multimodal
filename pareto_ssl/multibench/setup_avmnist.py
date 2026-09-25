#!/usr/bin/env python
"""
One-shot AV-MNIST organiser for the MultiBench λ-SSL benchmark.

You give it the downloaded archive (avmnist.tar.gz) OR an already-extracted folder,
and it lays the data out in EXACTLY the structure the benchmark loader expects,
then verifies the shapes. No guessing on your side.

Target layout (what pareto_ssl/multibench/image_backends.py reads):
    <dest>/image/train_data.npy      <dest>/image/test_data.npy
    <dest>/audio/train_data.npy      <dest>/audio/test_data.npy
    <dest>/train_labels.npy          <dest>/test_labels.npy

<dest> defaults to the "avmnist" path in CoMM/dataset/catalog.json — i.e. the path
the benchmark already looks up — so after this runs, `--dataset avmnist` just works.

Usage
-----
    # from an archive you downloaded (gdown 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp -O avmnist.tar.gz)
    python pareto_ssl/multibench/setup_avmnist.py --archive avmnist.tar.gz

    # or from a folder you already extracted somewhere
    python pareto_ssl/multibench/setup_avmnist.py --src /path/to/extracted/avmnist

    # override the destination (e.g. your Jean-Zay data root)
    python pareto_ssl/multibench/setup_avmnist.py --archive avmnist.tar.gz --dest <data root>/avmnist
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile

REQUIRED = [
    ("image", "train_data.npy"),
    ("image", "test_data.npy"),
    ("audio", "train_data.npy"),
    ("audio", "test_data.npy"),
    (None,    "train_labels.npy"),
    (None,    "test_labels.npy"),
]

# expected sample counts (MultiBench AV-MNIST)
EXPECT_N = {"train": 60000, "test": 10000}


def _rel(sub, name):
    return name if sub is None else os.path.join(sub, name)


def _default_dest():
    """The 'avmnist' path from CoMM/dataset/catalog.json (what the benchmark looks up)."""
    here = os.path.dirname(os.path.realpath(__file__))
    # pareto_ssl/multibench/ -> up to Program/ -> CoMM/dataset/catalog.json
    program = os.path.abspath(os.path.join(here, "..", ".."))
    catalog = os.path.join(program, "CoMM", "dataset", "catalog.json")
    with open(catalog) as f:
        return json.load(f)["avmnist"]["path"]


def _find_data_root(top):
    """Walk `top` and return the directory that contains image/train_data.npy."""
    for dirpath, _dirs, _files in os.walk(top):
        if os.path.isfile(os.path.join(dirpath, "image", "train_data.npy")):
            return dirpath
    return None


def _extract(archive, workdir):
    print(f"[1/4] Extracting {archive} ...")
    if not tarfile.is_tarfile(archive):
        sys.exit(f"ERROR: {archive} is not a tar archive. If it is a folder, use --src instead.")
    with tarfile.open(archive) as tf:
        tf.extractall(workdir)
    return workdir


def _place(src_root, dest):
    print(f"[3/4] Placing files into {dest}")
    os.makedirs(os.path.join(dest, "image"), exist_ok=True)
    os.makedirs(os.path.join(dest, "audio"), exist_ok=True)
    for sub, name in REQUIRED:
        s = os.path.join(src_root, _rel(sub, name))
        if not os.path.isfile(s):
            sys.exit(f"ERROR: expected file missing in source: {_rel(sub, name)}")
        d = os.path.join(dest, _rel(sub, name))
        shutil.copy2(s, d)
        print(f"        + {_rel(sub, name)}")


def _verify(dest):
    print(f"[4/4] Verifying shapes in {dest}")
    try:
        import numpy as np
    except ImportError:
        print("        (numpy not importable here — skipping shape check; files are in place)")
        return True
    ok = True
    for which in ("train", "test"):
        img = np.load(os.path.join(dest, "image", f"{which}_data.npy"), mmap_mode="r")
        aud = np.load(os.path.join(dest, "audio", f"{which}_data.npy"), mmap_mode="r")
        lab = np.load(os.path.join(dest, f"{which}_labels.npy"), mmap_mode="r")
        n = EXPECT_N[which]
        print(f"        {which}: image {tuple(img.shape)}  audio {tuple(aud.shape)}  labels {tuple(lab.shape)}")
        # image flattens to 28*28=784, audio to 112*112
        if img.shape[0] != n or aud.shape[0] != n or lab.shape[0] != n:
            print(f"        !! expected {n} samples for '{which}'")
            ok = False
        if int(np.prod(img.shape[1:])) != 28 * 28:
            print(f"        !! image feature size {int(np.prod(img.shape[1:]))} != 784")
            ok = False
        if int(np.prod(aud.shape[1:])) != 112 * 112:
            print(f"        !! audio feature size {int(np.prod(aud.shape[1:]))} != 12544")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="Organise AV-MNIST into the benchmark's expected layout.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive", help="Path to avmnist.tar.gz")
    g.add_argument("--src", help="Path to an already-extracted avmnist folder")
    ap.add_argument("--dest", default=None,
                    help="Destination root (default: 'avmnist' path in catalog.json)")
    args = ap.parse_args()

    dest = args.dest or _default_dest()
    print(f"Destination (catalog 'avmnist' path): {dest}\n")

    tmp = None
    if args.archive:
        tmp = tempfile.mkdtemp(prefix="avmnist_")
        top = _extract(args.archive, tmp)
    else:
        top = args.src
        print(f"[1/4] Using extracted source: {top}")

    print("[2/4] Locating the folder that contains image/train_data.npy ...")
    src_root = _find_data_root(top)
    if src_root is None:
        sys.exit("ERROR: could not find image/train_data.npy anywhere in the source. "
                 "Is this really the AV-MNIST archive?")
    print(f"        found: {src_root}")

    _place(src_root, dest)
    ok = _verify(dest)

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if ok:
        print("DONE — AV-MNIST is in place and shapes look correct.")
        print("You can now launch with:  --dataset avmnist")
    else:
        print("PLACED, but the shape check flagged something above — review before launching.")
        sys.exit(1)


if __name__ == "__main__":
    main()
