"""Download the MPI-Leipzig LEMON resting EEG (raw BrainVision) and reduce
each recording to Welch spectra per condition.

About 215 participants in two age groups (20-35 and 59-77; ages in 5-year
bins), 61 EEG channels plus VEOG at 2500 Hz referenced to FCz. Sixteen
alternating 60-s blocks of eyes closed (markers S210, one every 2 s) and
eyes open (S200). Each block's first 2 s are dropped. Preprocessing and
segment rejection as in psd_utils (rejection judged on EEG channels only, so
blinks on VEOG do not remove eyes-open data). One .npz per participant holds,
for ec and eo, the mean spectrum of every channel (VEOG included) over
kept segments and odd/even halves, per-segment spectra for posterior,
frontopolar, temporal and VEOG groups, and each segment's block index.

Output: $LEMON_OUT (default ./lemon_psd). Finished files are skipped.

Usage: python lemon_extract_psd.py [--workers 4] [--limit N]
"""
import argparse
import csv
import io
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import psd_utils as U

BASE = "https://fcp-indi.s3.amazonaws.com"
PREFIX = "data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/"
PART = f"{BASE}/data/Projects/INDI/MPI-LEMON/Participants_MPILMBB_LEMON.csv"
OUT = os.environ.get("LEMON_OUT", "lemon_psd")
TMP = os.environ.get("LEMON_TMP", "lemon_tmp")
GROUPS = {"post": ["O1", "Oz", "O2", "PO3", "POz", "PO4"],
          "fp": ["Fp1", "Fp2"],
          "temp": ["T7", "T8", "FT7", "FT8", "TP7", "TP8"],
          "veog": ["VEOG"]}
CODES = {"ec": "S210", "eo": "S200"}
SKIP_SEC = 2.0
# cleaned recordings (clean_adult.py: <sub>.set, the FCz reference restored)
# read instead of the downloads; FCz is dropped again so the average
# reference is the uncleaned one's
CLEANED = os.environ.get("LEMON_CLEANED", "")
REF = "FCz"


def fetch(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == tries - 1:
                return False
            time.sleep(2 ** k)
        except Exception:
            if k == tries - 1:
                return False
            time.sleep(2 ** k)
    return False


def subjects():
    out, tok = [], None
    while True:
        url = f"{BASE}/?list-type=2&prefix={PREFIX}&delimiter=/&max-keys=1000"
        if tok:
            url += "&continuation-token=" + urllib.request.quote(tok)
        x = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        ns = {"s": x.tag.split("}")[0].strip("{")}
        out += [c.find("s:Prefix", ns).text.rstrip("/").split("/")[-1]
                for c in x.findall("s:CommonPrefixes", ns)]
        if x.find("s:IsTruncated", ns).text != "true":
            return out
        tok = x.find("s:NextContinuationToken", ns).text


def participants():
    txt = urllib.request.urlopen(PART, timeout=120).read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(txt)))
    return {r[0]: dict(sex={"1": "F", "2": "M"}.get(r[1], ""), age_bin=r[2])
            for r in rows[1:] if r}


def spans(raw, code, srate):
    """(start, stop) samples of each block: runs of 2-s markers with this code."""
    on = sorted(o for o, d in zip(raw.annotations.onset, raw.annotations.description)
                if d.strip().endswith(code))
    blocks, cur = [], []
    for t in on:
        if cur and t - cur[-1] > 3.0:
            blocks.append(cur)
            cur = []
        cur.append(t)
    if cur:
        blocks.append(cur)
    return [(int((b[0] + SKIP_SEC) * srate), int((b[-1] + 2.0) * srate))
            for b in blocks if len(b) >= 5]


def stage(sub, dest):
    """Fetch the BrainVision files to dest/<sub>.vhdr/.vmrk/.eeg; returns the
    paths, or a 'missing <ext>' status."""
    stem = f"{BASE}/{PREFIX}{sub}/RSEEG/{sub}"
    paths = {e: os.path.join(dest, f"{sub}.{e}") for e in ("vhdr", "vmrk", "eeg")}
    for e, p in paths.items():
        if not fetch(f"{stem}.{e}", p):
            for q in paths.values():
                if os.path.exists(q):
                    os.remove(q)
            return f"missing {e}"
    # the headers name the original files; point them at the local ones
    for e in ("vhdr", "vmrk"):
        txt = open(paths[e], encoding="latin-1").read()
        txt = re.sub(r"^DataFile=.*$", f"DataFile={sub}.eeg", txt, flags=re.M)
        txt = re.sub(r"^MarkerFile=.*$", f"MarkerFile={sub}.vmrk", txt, flags=re.M)
        open(paths[e], "w", encoding="latin-1").write(txt)
    return paths


def one(args):
    sub, meta = args
    outfile = os.path.join(OUT, f"{sub}.npz")
    if os.path.exists(outfile):
        return sub, "cached"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    paths = {}
    try:
        import mne
        if CLEANED:
            path = os.path.join(CLEANED, f"{sub}.set")
            if not os.path.exists(path):
                return sub, "missing set"
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
            raw.drop_channels([REF])
        else:
            paths = stage(sub, TMP)
            if isinstance(paths, str):
                return sub, paths
            raw = mne.io.read_raw_brainvision(paths["vhdr"], preload=True, verbose="ERROR")
        if "VEOG" in raw.ch_names:
            raw.set_channel_types({"VEOG": "eog"})
        raw.pick(["eeg", "eog"])
        x, srate, names = U.preprocess(raw)
        eeg_mask = np.array([n != "VEOG" for n in names])
        save, status = {}, []
        for key, code in CODES.items():
            sp = spans(raw, code, srate)
            P, f, src = U.welch_segments(x, srate, sp)
            if P is None:
                status.append(f"{key}:none")
                continue
            s, fr = U.summarise(P, f, U.reject(P, eeg_mask), GROUPS, names)
            for k2, v in s.items():
                save[f"{key}_{k2}"] = v
            save[f"{key}_block"] = src
            save[f"{key}_n_blocks"] = len(sp)
            status.append(f"{key}:ok")
        if not save:
            return sub, ";".join(status)
        lo, hi = (float(v) for v in meta.get("age_bin", "nan-nan").split("-")) \
            if "-" in meta.get("age_bin", "") else (np.nan, np.nan)
        np.savez_compressed(outfile, freqs=fr.astype(np.float32), ch_names=np.array(names),
                            subject=sub, age_bin=meta.get("age_bin", ""),
                            age=(lo + hi) / 2, sex=meta.get("sex", ""),
                            status=";".join(status), **save)
        return sub, ";".join(status)
    except Exception as e:
        return sub, f"error {type(e).__name__}: {e}"
    finally:
        for p in (paths.values() if isinstance(paths, dict) else ()):
            try:
                os.remove(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    meta = participants()
    subs = subjects()
    if a.limit:
        subs = subs[:a.limit]
    print(f"{len(subs)} participants queued", flush=True)
    t0 = time.time()
    counts = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, (s, meta.get(s, {}))) for s in subs]
        for k, fu in enumerate(as_completed(futs)):
            sub, st = fu.result()
            ok = st == "cached" or st == "ec:ok;eo:ok"
            counts["complete" if ok else "other"] = counts.get("complete" if ok else "other", 0) + 1
            if (k + 1) % 10 == 0 or not ok:
                print(f"[{k + 1}/{len(subs)}] {(time.time() - t0) / 60:6.1f} min {sub} {st} {counts}",
                      flush=True)
    print(f"FINISHED {len(subs)} in {(time.time() - t0) / 60:.1f} min {counts}", flush=True)


if __name__ == "__main__":
    main()
