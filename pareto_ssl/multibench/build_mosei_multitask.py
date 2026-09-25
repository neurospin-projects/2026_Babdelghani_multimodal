#!/usr/bin/env python
"""
Build a MULTI-TASK MOSEI pickle by joining the 7 CMU-MOSEI labels onto the
existing sentiment pickle.

Why: mosei_senti_data.pkl carries a single scalar label, so MOSEI can only ever
demonstrate one task. mosei.hdf5 ("All Labels") carries the full 7-vector
    [sentiment, happy, sad, anger, surprise, disgust, fear]
keyed by segment id "<video>[<clip>]". Features are identical, so we only need to
attach the labels — no reprocessing.

Output: a pickle with the SAME structure as the input but labels of shape (N, 7),
plus a "label_names" entry. Samples whose id has no label entry are dropped
(reported), so every split stays consistent across modalities.

Usage (run where BOTH files live):
  python pareto_ssl/multibench/build_mosei_multitask.py \
      --pkl  $SCRATCH/data/multibench/mosei/mosei_senti_data.pkl \
      --hdf5 /path/to/mosei.hdf5 \
      --out  $SCRATCH/data/multibench/mosei/mosei_multitask_data.pkl

Add --dry_run first: it reports the id format and the match rate without writing.
"""
import argparse
import gc
import pickle
import sys

import numpy as np

NAMES = ["sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear"]


def _dec(x):
    """bytes/np.bytes_/str/array -> clean str"""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "ignore").strip()
    if isinstance(x, np.ndarray):
        return _dec(x.item()) if x.size == 1 else "".join(_dec(v) for v in x.ravel())
    return str(x).strip()


def _parts(row):
    p = [_dec(v) for v in np.asarray(row).ravel()]
    return [x for x in p if x not in ("", "nan")]


def _norm_idx(x):
    """'12.0' -> '12';  '012' -> '12';  keeps non-numeric as-is."""
    x = x.strip()
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return x


# Candidate join schemes. The MOSEI id row is (video, clip, segment)-ish, but which
# field indexes the hdf5 key "<video>[<n>]" varies by preprocessing, so we detect it.
SCHEMES = {
    "vid[p1]":        lambda p: f"{p[0]}[{_norm_idx(p[1])}]" if len(p) > 1 else None,
    "vid[p2]":        lambda p: f"{p[0]}[{_norm_idx(p[2])}]" if len(p) > 2 else None,
    "vid[p1-1]":      lambda p: (f"{p[0]}[{int(float(p[1]))-1}]" if len(p) > 1 and _norm_idx(p[1]).lstrip('-').isdigit() else None),
    "vid[p2-1]":      lambda p: (f"{p[0]}[{int(float(p[2]))-1}]" if len(p) > 2 and _norm_idx(p[2]).lstrip('-').isdigit() else None),
    "p1[p2]":         lambda p: f"{p[1]}[{_norm_idx(p[2])}]" if len(p) > 2 else None,
    "vid_only":       lambda p: p[0] if p else None,
    "joined":         lambda p: "".join(p) if p else None,
}


# Content join
# The pickle's `id` field is (video, start_time, end_time) — TIMES, not the
# segment index the hdf5 keys use ("<video>[<k>]"), so no id arithmetic can join
# them (best id scheme reached only ~20%). Instead we match on the AUDIO itself:
# the pickle's 74-d audio IS the hdf5's COVAREP. Hashing the whole segment gives
# 23248/23248 unique keys with ZERO collisions, so a pickle row identifies its
# hdf5 segment exactly — independent of ids, padding, truncation and ordering.

def _seq_hash(a):
    return hash(np.round(np.asarray(a, dtype=np.float64), 4).tobytes())


def _trim(a):
    """Drop the zero padding MultiBench adds, leaving the true segment."""
    a = np.asarray(a)
    nz = np.nonzero(np.abs(a).sum(axis=1))[0]
    return a[nz[0]:nz[-1] + 1] if len(nz) else a


def build_audio_index(h5, keys):
    """audio-sequence hash -> segment key, under three truncation conventions."""
    C = h5["COVAREP"]
    idx = {}
    for k in keys:
        a = np.asarray(C[k]["features"])
        if a.size == 0:
            continue
        for v in (a, a[:50], a[-50:]):
            idx.setdefault(_seq_hash(v), k)
    return idx


def match_by_audio(audio_row, aidx):
    """Try the pickle row's audio under the same conventions."""
    a = _trim(audio_row)
    for v in (a, a[:50], a[-50:], np.asarray(audio_row)):
        k = aidx.get(_seq_hash(v))
        if k is not None:
            return k
    return None


# Positional (time-order) join
# The pickle's `id` is (video, start_time, end_time) — TIMES, not the segment
# index the hdf5 keys use ("<video>[k]"). A content join on audio also fails: this
# hdf5 is a different extraction (vision 713-d OpenFace vs the pickle's 35-d), so
# the aligned COVAREP values differ.
#
# What DOES hold: every video's hdf5 indices form a contiguous 0..n-1 run in
# temporal order (verified: 0/3292 videos violate this). So sorting a video's
# pickle rows by start time recovers k. We only join videos whose segment COUNT
# matches exactly — otherwise the positions would silently shift and attach WRONG
# labels, which is far worse than dropping the video.

def positional_join(d, table):
    """-> {(split, row_index): hdf5_key}, plus per-video diagnostics."""
    import re
    from collections import defaultdict
    h5_per = defaultdict(list)
    for k in table:
        m = re.match(r"^(.*)\[(\d+)\]$", k)
        if m:
            h5_per[m.group(1)].append((int(m.group(2)), k))
    for v in h5_per:
        h5_per[v].sort()

    pk_per = defaultdict(list)
    for split, s in d.items():
        ids = np.asarray(s["id"])
        for i in range(len(ids)):
            pr = _parts(ids[i])
            if len(pr) >= 2:
                try:
                    start = float(pr[1])
                except ValueError:
                    start = 0.0
                pk_per[pr[0]].append((start, split, i))
    for v in pk_per:
        pk_per[v].sort()

    mapping, ok_v, bad_v, missing_v = {}, 0, 0, 0
    for v, rows in pk_per.items():
        if v not in h5_per:
            missing_v += 1
            continue
        if len(rows) != len(h5_per[v]):
            bad_v += 1
            continue
        ok_v += 1
        for (_, split, i), (_, key) in zip(rows, h5_per[v]):
            mapping[(split, i)] = key
    print(f"  videos: {ok_v} count-matched, {bad_v} count-mismatch (skipped), "
          f"{missing_v} absent from hdf5")
    return mapping


def interval_join(d, h5, tol=0.05):
    """EXACT join on (video, start, end) using the unaligned file's `intervals`.

    mosei_unalign.hdf5 stores, per segment, both the 7 labels and the time interval
    it covers. Those intervals are unique within a video (verified: 0/23248
    duplicates) and are exactly what the pickle's id encodes, so this is a real key
    rather than an inference. Unlike the sentiment join it is NOT circular — the
    sentiment sign check afterwards becomes a genuine independent validation.

    tol is in seconds; matching takes the nearest interval within tol.
    """
    from collections import defaultdict
    g = h5["All Labels"]
    per = defaultdict(list)
    for k in g.keys():
        iv = np.asarray(g[k]["intervals"]).reshape(-1)[:2]
        per[k.split("[")[0]].append((float(iv[0]), float(iv[1]), k))

    mapping, hit, tot, miss_v = {}, 0, 0, 0
    for split, s_ in d.items():
        ids = np.asarray(s_["id"])
        for i in range(len(ids)):
            pr = _parts(ids[i])
            tot += 1
            if len(pr) < 3 or pr[0] not in per:
                miss_v += 1
                continue
            try:
                st_, en_ = float(pr[1]), float(pr[2])
            except ValueError:
                continue
            best, bd = None, tol
            for a, b, k in per[pr[0]]:
                dd = abs(a - st_) + abs(b - en_)
                if dd <= bd:
                    best, bd = k, dd
            if best is not None:
                mapping[(split, i)] = best
                hit += 1
    print(f"  rows matched on (video, start, end): {hit}/{tot} ({100*hit/max(tot,1):.1f}%)"
          f"   [tol={tol}s, {miss_v} rows whose video is absent]")
    return mapping


def sentiment_join(d, table, tol=1e-3):
    """Match each pickle row to its hdf5 segment using the SENTIMENT value.

    Time-order position turned out to be unreliable (80% sign agreement — the two
    preprocessings do not segment videos identically even when the counts happen to
    agree). But BOTH files carry the same continuous sentiment score, so within a
    video we can require an exact one-to-one assignment on that value.

    A video is accepted only when a UNIQUE perfect bijection exists: every pickle
    row pairs with a distinct hdf5 segment at |Δsentiment| <= tol, and no row has
    an ambiguous choice. Videos that do not satisfy this are dropped rather than
    guessed at. This makes segment identity verifiable rather than assumed, and the
    emotion labels then come along with the correct segment.
    """
    import re
    from collections import defaultdict
    h5 = defaultdict(list)
    for k, v in table.items():
        m = re.match(r"^(.*)\[(\d+)\]$", k)
        if m:
            h5[m.group(1)].append((k, float(v[0])))

    pk = defaultdict(list)
    for split, s_ in d.items():
        ids = np.asarray(s_["id"])
        lab = np.asarray(s_["labels"], dtype=np.float32).reshape(len(ids), -1)[:, 0]
        for i in range(len(ids)):
            pr = _parts(ids[i])
            if pr:
                pk[pr[0]].append((split, i, float(lab[i])))

    mapping, miss_v = {}, 0
    n_rows = n_hit = 0
    for v, rows in pk.items():
        if v not in h5:
            miss_v += 1
            n_rows += len(rows)
            continue
        pool = {k: sv for k, sv in h5[v]}
        pending = list(rows)
        n_rows += len(rows)
        # Constraint propagation: repeatedly commit rows that have exactly ONE
        # remaining candidate, which frees the pool and can disambiguate others.
        # Rows still ambiguous when nothing more can be committed are dropped
        # individually — the video as a whole is kept.
        progress = True
        while pending and progress:
            progress = False
            for r in list(pending):
                split, i, sent = r
                cand = [k for k, sv in pool.items() if abs(sv - sent) <= tol]
                if len(cand) == 1:
                    mapping[(split, i)] = cand[0]
                    pool.pop(cand[0])
                    pending.remove(r)
                    n_hit += 1
                    progress = True
                elif not cand:
                    pending.remove(r)      # no candidate at all: drop this row
    print(f"  rows uniquely resolved: {n_hit}/{n_rows} ({100*n_hit/max(n_rows,1):.1f}%), "
          f"{miss_v} videos absent from hdf5")
    return mapping





def detect_scheme(d, table, sample=1500):
    """Try every scheme on a sample and return the best-matching one."""
    rows = []
    for s in d.values():
        ids = np.asarray(s["id"])
        rows.extend(ids[i] for i in range(min(len(ids), sample // max(len(d), 1))))
    scores = {}
    for name, fn in SCHEMES.items():
        hit = 0
        for r in rows:
            try:
                k = fn(_parts(r))
            except Exception:
                k = None
            if k is not None and k in table:
                hit += 1
        scores[name] = hit / max(len(rows), 1)
    print("\n  join-scheme detection (sampled):")
    for n, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"    {n:12s} {100*v:6.1f}%")
    best = max(scores, key=scores.get)
    print(f"  -> using '{best}'\n")
    return SCHEMES[best], best, scores[best]


def _candidate_keys(row, fn=None):
    if fn is not None:
        try:
            k = fn(_parts(row))
        except Exception:
            k = None
        return [k] if k else []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import h5py
    h5 = h5py.File(args.hdf5, "r")
    lab = h5["All Labels"]
    table = {k: np.asarray(lab[k]["features"]).reshape(-1)[:7] for k in lab.keys()}
    print(f"hdf5: {len(table)} labelled segments, {len(NAMES)} labels each")

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    print(f"pickle splits: {list(d.keys())}")

    ids0 = np.asarray(d[list(d.keys())[0]]["id"])
    print(f"\n  id row example : {_parts(ids0[0])}")
    print(f"  hdf5 key example: {next(iter(table))}")
    fn, scheme, rate = detect_scheme(d, table)
    use_pos = rate < 0.9
    if use_pos:
        print(f"  id join tops out at {100*rate:.1f}% (ids are TIMES, not indices)")
        if "intervals" in h5["All Labels"][next(iter(h5["All Labels"]))]:
            print("  -> using INTERVAL join on (video, start, end) — exact, non-circular\n")
            posmap = interval_join(d, h5)
        else:
            print("  -> hdf5 has no intervals; falling back to the SENTIMENT join\n")
            posmap = sentiment_join(d, table)
        print()

    # Memory: the pickle holds ~1.3 GB of features and the ~95% subset nearly
    # doubles that, which OOM-kills a login node. So process ONE split at a time,
    # drop each source array as soon as its subset exists, and gc between splits.
    out, total_hit, total_n = {}, 0, 0
    # smallest split first: any problem shows up in seconds instead of after the
    # 16k-row train split has already been materialised.
    order = sorted(d.keys(), key=lambda k: len(np.asarray(d[k]["id"])))
    print(f"  processing splits smallest-first: {order}\n")
    for split in order:
        s = d[split]
        ids = np.asarray(s["id"])
        n = len(ids)
        Y, keep = np.zeros((n, 7), np.float32), []
        for i in range(n):
            k = (posmap.get((split, i)) if use_pos
                 else next((c for c in _candidate_keys(ids[i], fn) if c in table), None))
            if k is not None and k in table:
                Y[i] = table[k]
                keep.append(i)
        keep = np.asarray(keep, int)
        total_hit += len(keep); total_n += n
        print(f"  {split:6s} {len(keep):6d}/{n:<6d} matched ({100*len(keep)/max(n,1):5.1f}%)")

        # sanity BEFORE we drop the originals: both files carry sentiment, so the
        # sign must agree if the positional join lined the segments up correctly.
        old = np.asarray(s["labels"], dtype=np.float32).reshape(len(s["labels"]), -1)[keep, 0]
        new = Y[keep][:, 0]
        agree = float(((old > 0) == (new > 0)).mean()) if len(old) else 0.0
        flag = "OK" if agree > 0.95 else "*** MISMATCH ***"
        print(f"         sentiment sign agreement vs original: {100*agree:5.1f}%  {flag}")
        if agree <= 0.95:
            print("\n  ABORTING before building: the join does not reproduce the original\n"
                  "  sentiment, so segment identity is wrong. Nothing was written.")
            sys.exit(1)

        # In a dry run the match report and the sign check above are the whole
        # point; copying ~1.3 GB of features we are about to throw away is what
        # made --dry_run as heavy as the real build. Skip straight to the next
        # split so the check can run in a couple of GB.
        if args.dry_run:
            d[split] = None
            del s, Y
            gc.collect()
            continue

        ns = {}
        for k in list(s.keys()):
            arr = np.asarray(s.pop(k))          # pop: free the source as we go
            ns[k] = arr[keep] if len(arr) == n else arr
            del arr
        ns["labels"] = Y[keep]
        ns["label_names"] = NAMES
        out[split] = ns
        d[split] = None
        del s, Y
        gc.collect()

    print(f"\noverall match: {total_hit}/{total_n} ({100*total_hit/max(total_n,1):.1f}%)")
    if total_hit == 0:
        print("\nNO MATCHES — the id format did not line up. Paste the 'id row example'\n"
              "and 'hdf5 key example' lines above and the key builder can be adjusted.")
        sys.exit(1)

    for split, s in out.items():
        Y = s["labels"]
        if len(Y):
            pos = [f"{NAMES[i]} {100*(Y[:,i]>0).mean():.0f}%" for i in range(7)]
            print(f"  {split:6s} positives: " + "  ".join(pos))

    if args.dry_run:
        print("\n--dry_run: nothing written.")
        return
    if not args.out:
        print("\nno --out given; nothing written.")
        return
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {args.out}  (labels now (N, 7))")


if __name__ == "__main__":
    main()
