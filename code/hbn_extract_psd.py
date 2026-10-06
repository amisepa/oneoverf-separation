"""Download HBN-EEG RestingState recordings and reduce each to Welch PSDs.

For every subject: fetch the .set file from the OpenNeuro S3 mirror (or,
where the mirror serves it only by version ID, as in ds005516, from the
versioned URL given by the OpenNeuro API), split
the recording into eyes-open and eyes-closed blocks from the instruction
markers (dropping the first 2 s after each cue), and compute per-channel
Welch PSDs per condition (4 s Hann, 50% overlap), plus odd/even segment
splits, three interleaved thirds of the segments (kept segment index mod 3)
and per-block PSDs. For the posterior and temporal channels the spectrum of
every single segment is stored as well, with the rejection statistic and the
block of each segment, so that any other split or rejection rule can be
formed without the recordings.

One .npz is written per subject (under a temporary name first, so an
interrupted run leaves no partial file); subjects already done are skipped,
so the script can be restarted. A file that is absent from the mirror is
checked against the file listing of the OpenNeuro snapshot before the
subject is counted as having no resting recording. Network errors are
retried, and subjects that still fail are run again in up to --passes
further passes; they are never counted as missing. Counts of queued, done,
absent and failed subjects are printed per dataset, and the outcome of every
subject is written to $HBN_OUT/extract_status.csv.

Output directory: $HBN_OUT (default ./hbn_psd).

Usage:
    python hbn_extract_psd.py ds005505 [ds005506 ...] --workers 6 [--limit N]
                              [--passes 3] [--subjects FILE]
  --subjects  text file with one "dataset subject" pair per line; only these
              are extracted
"""
import argparse
import csv
import functools
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

S3 = "https://s3.amazonaws.com/openneuro.org"
GRAPHQL = "https://openneuro.org/crn/graphql"
OUT = os.environ.get("HBN_OUT", "hbn_psd")
TMP = os.environ.get("HBN_TMP", "hbn_tmp")

# analysis settings
SRATE_TARGET = 250.0      # decimate; we only fit 2-45 Hz
WIN_SEC = 4.0
OVERLAP = 0.5
FRANGE = (1.0, 60.0)      # stored range; fits use a subrange
EO_SKIP, EO_LEN = 2.0, 17.0   # s after the open-eyes cue
EC_SKIP, EC_LEN = 2.0, 37.0   # s after the close-eyes cue
AMP_REJECT = 250.0        # uV, peak-to-peak per 4 s segment per channel
# posterior cluster, GSN-HydroCel-129; per-block spectra are stored for these
ROI = ["E70", "E75", "E83", "E62", "E65", "E90"]
# temporal cluster (muscle-sensitive flank band of hbn_topography_flanks.py)
TEMPORAL = ["E116", "E117", "E111", "E123", "E34", "E35", "E40", "E109"]
TRIES = 6                 # attempts per request, waiting 2, 4, ... 32 s between


class Transient(IOError):
    """A request that kept failing for a reason other than a missing file."""


def _retry(what, url, tries=TRIES):
    """Call what(); None if the file does not exist (HTTP 404), Transient if
    every attempt failed otherwise. A network error must never be read as a
    missing file: that drops the subject without a trace."""
    err = None
    for k in range(tries):
        try:
            return what()
        except urllib.error.HTTPError as e:
            if e.code == 404:                        # missing is not transient
                return None
            err = e
        except Exception as e:
            err = e
        if k < tries - 1:
            time.sleep(2 ** (k + 1))
    raise Transient(f"{type(err).__name__}: {err} ({url})")


def fetch(url, dest):
    """Download url to dest. True if done, False if the file does not exist."""
    def get():
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
            want = r.headers.get("Content-Length")
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        # a connection closed early ends the read without an error
        if want is not None and got != int(want):
            raise IOError(f"incomplete download, {got} of {want} bytes")
        return True
    return bool(_retry(get, url))


def read_tsv(url):
    """Rows of a remote .tsv, or None if the file does not exist."""
    def get():
        with urllib.request.urlopen(url, timeout=120) as r:
            txt = r.read().decode("utf-8", "replace")
        return list(csv.DictReader(io.StringIO(txt), delimiter="\t"))
    return _retry(get, url)


def graphql(query):
    body = json.dumps({"query": query}).encode()

    def get():
        req = urllib.request.Request(GRAPHQL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        if not d.get("data"):
            raise IOError(f"GraphQL: {str(d.get('errors'))[:200]}")
        return d["data"]
    d = _retry(get, GRAPHQL)
    if d is None:
        raise Transient("GraphQL endpoint returned 404")
    return d


@functools.lru_cache(maxsize=None)
def snapshot_tree(ds, tree=""):
    """{filename: (id, urls)} for one directory of the latest snapshot.

    Raises Transient when the listing cannot be read; a failure is not
    cached, so the next subject asks again."""
    d = graphql(f'{{ dataset(id: "{ds}") {{ latestSnapshot {{ tag }} }} }}')
    tag = d["dataset"]["latestSnapshot"]["tag"]
    arg = f'(tree: "{tree}")' if tree else ""
    d = graphql(f'{{ snapshot(datasetId: "{ds}", tag: "{tag}") '
                f'{{ files{arg} {{ id filename urls }} }} }}')
    return {f["filename"]: (f["id"], f["urls"]) for f in d["snapshot"]["files"]}


def snapshot_url(ds, sub, filename):
    """Versioned download URL of sub/eeg/filename; None if the snapshot does
    not list the file."""
    top = snapshot_tree(ds)
    if sub not in top:
        return None
    eeg = snapshot_tree(ds, top[sub][0]).get("eeg")
    if not eeg:
        return None
    f = snapshot_tree(ds, eeg[0]).get(filename)
    return f[1][0] if f and f[1] else None


def participants(ds):
    rows = read_tsv(f"{S3}/{ds}/participants.tsv")
    if rows is None:
        raise Transient(f"{ds}: participants.tsv not found")
    out = {}
    for r in rows:
        pid = r.get("participant_id")
        if not pid:
            continue
        try:
            age = float(r.get("age", "nan"))
        except ValueError:
            age = float("nan")
        out[pid] = dict(age=age, sex=r.get("sex", ""),
                        resting=r.get("RestingState", ""),
                        release=r.get("release_number", ""))
    return out


def welch(x, srate, nper, noverlap):
    """One-sided PSD in uV^2/Hz per segment; returns (nseg, nch, nfreq)."""
    step = nper - noverlap
    nseg = 1 + (x.shape[1] - nper) // step if x.shape[1] >= nper else 0
    if nseg <= 0:
        return None, None
    win = np.hanning(nper + 1)[:nper]
    scale = 1.0 / (srate * np.sum(win ** 2))
    f = np.fft.rfftfreq(nper, 1.0 / srate)
    out = np.empty((nseg, x.shape[0], f.size), dtype=np.float32)
    for s in range(nseg):
        seg = x[:, s * step: s * step + nper]
        seg = seg - seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg * win, axis=1)
        p = (np.abs(X) ** 2) * scale * 2.0
        p[:, 0] /= 2.0
        if nper % 2 == 0:
            p[:, -1] /= 2.0
        out[s] = p
    return out, f


def blocks_psd(raw_data, srate, spans, nper, noverlap, roi_ix=None, seg_ix=None):
    """Welch segments over a list of (start, stop) sample spans.

    Returns the mean PSD, the odd/even segment split means, and -- for the ROI
    channels -- one PSD per BLOCK. The per-block spectra support the
    within-condition identification route, in which lambda is estimated from
    spontaneous block-to-block fluctuation in the aperiodic background rather
    than from a manipulation that moves the background and the oscillation at
    the same time.

    The sixth value is a dict: thirds (3, nch, nfreq), the means of the kept
    segments k, k + 3, ... for k = 0, 1, 2 (nan where a third is empty) and
    n_thirds, their segment counts; for every segment before rejection, in
    time order, seg (the spectra of the channels seg_ix), seg_block (index of
    its span), seg_ratio (the rejection statistic) and seg_used (whether it
    entered the means).
    """
    segs, per_block, block_of = [], [], []
    for k, (a, b) in enumerate(spans):
        if b - a < nper:
            continue
        p, f = welch(raw_data[:, a:b], srate, nper, noverlap)
        if p is not None:
            segs.append(p)
            block_of += [k] * p.shape[0]
            if roi_ix is not None:
                per_block.append(p[:, roi_ix, :].mean(axis=(0, 1)))
    if not segs:
        return None, None, None, None, None, None
    P = np.concatenate(segs, axis=0)
    # amplitude-based segment rejection: drop segments whose broadband power
    # is a gross outlier in any channel
    med = np.median(P, axis=0, keepdims=True)
    ratio = np.max(P / np.maximum(med, 1e-12), axis=(1, 2))
    keep = ratio < 50.0
    used = keep if keep.sum() >= 4 else np.ones(P.shape[0], bool)
    extra = dict(seg=P[:, seg_ix, :] if seg_ix is not None else None,
                 seg_block=np.asarray(block_of, np.int16),
                 seg_ratio=ratio.astype(np.float32), seg_used=used)
    if keep.sum() >= 4:
        P = P[keep]
    thirds = np.full((3,) + P.shape[1:], np.nan, np.float32)
    for k in range(3):
        if P[k::3].shape[0]:
            thirds[k] = P[k::3].mean(0)
    extra.update(thirds=thirds,
                 n_thirds=np.array([P[k::3].shape[0] for k in range(3)], np.int16))
    _, f = welch(raw_data[:, spans[0][0]:spans[0][0] + nper], srate, nper, noverlap)
    blocks = np.asarray(per_block, dtype=np.float32) if per_block else None
    return P.mean(0), P[0::2].mean(0), P[1::2].mean(0), (f, P.shape[0]), blocks, extra


def process(args):
    ds, sub, meta = args
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    outfile = os.path.join(OUT, f"{ds}_{sub}.npz")
    if os.path.exists(outfile):
        return sub, "cached"

    base = f"{S3}/{ds}/{sub}/eeg/{sub}_task-RestingState"
    setf = os.path.join(TMP, f"{sub}_RestingState.set")
    try:
        # A file missing under its plain key may be served by version ID only
        # (ds005516), so absence is decided by the snapshot's file listing.
        ev = read_tsv(base + "_events.tsv")
        if ev is None:
            url = snapshot_url(ds, sub, f"{sub}_task-RestingState_events.tsv")
            if not url:
                return sub, "no-events"
            ev = read_tsv(url)
            if ev is None:
                return sub, "fetch-failed: events listed in the snapshot but not served"
        if not ev:
            return sub, "empty-events"
        if not fetch(base + "_eeg.set", setf):
            url = snapshot_url(ds, sub, f"{sub}_task-RestingState_eeg.set")
            if not url:
                return sub, "no-set-file"
            if not fetch(url, setf):
                return sub, "fetch-failed: .set listed in the snapshot but not served"
    except Transient as e:
        try:
            os.remove(setf)
        except OSError:
            pass
        return sub, f"fetch-failed: {e}"

    try:
        import mne
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")
            raw.pick("eeg")
            srate0 = raw.info["sfreq"]
            dec = max(1, int(round(srate0 / SRATE_TARGET)))
            raw.filter(0.4, min(SRATE_TARGET / 2 - 5, 100.0), fir_design="firwin",
                       verbose="ERROR")
            if dec > 1:
                raw.resample(srate0 / dec, verbose="ERROR")
            srate = raw.info["sfreq"]
            raw.set_eeg_reference("average", verbose="ERROR")
            X = raw.get_data() * 1e6          # uV
            ch = list(raw.ch_names)

        nper = int(round(WIN_SEC * srate))
        nov = int(round(OVERLAP * nper))

        def spans(cue, skip, length):
            out = []
            for r in ev:
                if r.get("value") != cue:
                    continue
                t0 = float(r["onset"]) + skip
                a, b = int(t0 * srate), int((t0 + length) * srate)
                if b <= X.shape[1]:
                    out.append((a, b))
            return out

        eo = spans("instructed_toOpenEyes", EO_SKIP, EO_LEN)
        ec = spans("instructed_toCloseEyes", EC_SKIP, EC_LEN)
        if len(eo) < 2 or len(ec) < 2:
            return sub, f"too-few-blocks eo={len(eo)} ec={len(ec)}"

        roi_ix = [ch.index(c) for c in ROI if c in ch]
        seg_names = [c for c in ROI + TEMPORAL if c in ch]
        seg_ix = [ch.index(c) for c in seg_names]
        res = {}
        for name, sp in (("eo", eo), ("ec", ec)):
            full, odd, even, info, blocks, extra = blocks_psd(
                X, srate, sp, nper, nov, roi_ix if len(roi_ix) >= 3 else None, seg_ix)
            if full is None:
                return sub, f"no-segments-{name}"
            res[name] = (full, odd, even, info, blocks, extra)
        f = res["eo"][3][0]
        keep = (f >= FRANGE[0]) & (f <= FRANGE[1])

        more = {}
        for name in ("eo", "ec"):
            x = res[name][5]
            more[f"{name}_thirds"] = x["thirds"][:, :, keep]
            more[f"n_seg_{name}_thirds"] = x["n_thirds"]
            more[f"{name}_seg"] = x["seg"][:, :, keep]
            for k in ("seg_block", "seg_ratio", "seg_used"):
                more[f"{name}_{k}"] = x[k]

        # written under a temporary name and renamed, so that a file with the
        # final name is always complete
        part = outfile + ".part"
        with open(part, "wb") as fh:
            np.savez_compressed(
                fh, seg_names=np.array(seg_names), **more,
                freqs=f[keep].astype(np.float32),
                ch_names=np.array(ch),
                eo=res["eo"][0][:, keep], eo_odd=res["eo"][1][:, keep],
                eo_even=res["eo"][2][:, keep],
                ec=res["ec"][0][:, keep], ec_odd=res["ec"][1][:, keep],
                ec_even=res["ec"][2][:, keep],
                n_seg_eo=res["eo"][3][1], n_seg_ec=res["ec"][3][1],
                eo_blocks=(res["eo"][4][:, keep] if res["eo"][4] is not None
                           else np.zeros((0, keep.sum()), np.float32)),
                ec_blocks=(res["ec"][4][:, keep] if res["ec"][4] is not None
                           else np.zeros((0, keep.sum()), np.float32)),
                roi_names=np.array([c for c in ROI if c in ch]),
                srate=srate, age=meta.get("age", np.nan), sex=meta.get("sex", ""),
                release=meta.get("release", ""), subject=sub, dataset=ds,
            )
        os.replace(part, outfile)
        return sub, "ok"
    except Exception as e:
        return sub, f"error: {type(e).__name__}: {e}"
    finally:
        for p in (setf, setf.replace(".set", ".fdt"), outfile + ".part"):
            try:
                os.remove(p)
            except OSError:
                pass


def retriable(status):
    """Outcomes a later pass may change: network failures, and errors while
    reading a recording (a damaged download reads as a corrupt file)."""
    return status.startswith(("fetch-failed", "error"))


def table(queued, last):
    """Per-dataset counts: queued, done, absent or unusable, failed, pending."""
    lines = [f"  {'dataset':10s} {'queued':>6s} {'done':>6s} {'absent':>6s} "
             f"{'failed':>6s} {'pending':>7s}"]
    for ds in sorted(queued):
        st = [last.get((ds, s)) for s in queued[ds]]
        done = sum(x in ("ok", "cached") for x in st)
        fail = sum(x is not None and retriable(x) for x in st)
        pend = sum(x is None for x in st)
        lines.append(f"  {ds:10s} {len(st):6d} {done:6d} {len(st) - done - fail - pend:6d} "
                     f"{fail:6d} {pend:7d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--passes", type=int, default=3,
                    help="further passes over the subjects that failed")
    ap.add_argument("--subjects", default="")
    a = ap.parse_args()

    only = {}
    if a.subjects:
        with open(a.subjects) as fh:
            for line in fh:
                if line.strip():
                    ds, sub = line.split()
                    only.setdefault(ds, set()).add(sub)
    datasets = a.datasets or sorted(only)

    # the participant table decides what is queued: if it cannot be read the
    # run stops, since going on would leave out the whole release
    jobs, queued = [], {}
    for ds in datasets:
        pp = participants(ds)
        subs = [s for s, m in pp.items() if m["resting"] in ("available", "caution", "")]
        n_flag = len(subs)
        if only:
            subs = [s for s in subs if s in only.get(ds, ())]
        if a.limit:
            subs = subs[:a.limit]
        queued[ds] = subs
        jobs += [(ds, s, pp[s]) for s in subs]
        print(f"{ds}: {len(pp)} participants, {n_flag} with a resting recording "
              f"flagged, {len(subs)} queued", flush=True)
    # mixed order: a run that stops early has lost no release in particular
    random.Random(0).shuffle(jobs)
    print(f"{len(jobs)} subjects queued across {len(datasets)} datasets", flush=True)

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    last = {}
    todo = jobs
    for npass in range(a.passes + 1):
        if npass:
            print(f"pass {npass + 1}: {len(todo)} failed subjects again", flush=True)
            time.sleep(60)
        counts = {}
        done = 0
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(process, j): j for j in todo}
            for fu in as_completed(futs):
                ds, sub = futs[fu][:2]
                try:
                    status = fu.result()[1]
                except Exception as e:              # a worker process died
                    status = f"error: worker {type(e).__name__}: {e}"
                last[ds, sub] = status
                key = status.split(":")[0].split(" ")[0]
                counts[key] = counts.get(key, 0) + 1
                done += 1
                if done % 10 == 0 or status not in ("ok", "cached"):
                    el = time.time() - t0
                    print(f"[{done}/{len(todo)}] {el/60:6.1f} min  {ds} {sub} -> {status}  "
                          f"{counts}", flush=True)
                if done % 200 == 0:
                    print(table(queued, last), flush=True)
        print(f"pass {npass + 1} done, {(time.time()-t0)/60:.1f} min  {counts}", flush=True)
        print(table(queued, last), flush=True)
        todo = [j for j in todo if retriable(last[j[0], j[1]])]
        if not todo:
            break

    with open(os.path.join(OUT, "extract_status.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "subject", "status"])
        for (ds, sub), status in sorted(last.items()):
            w.writerow([ds, sub, status])
    n_ok = sum(x in ("ok", "cached") for x in last.values())
    print(f"FINISHED {len(last)} subjects in {(time.time()-t0)/60:.1f} min: "
          f"{n_ok} done, {len(last) - n_ok - len(todo)} absent or unusable, "
          f"{len(todo)} FAILED", flush=True)
    if todo:
        print("failed subjects remain: run the same command again", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
