"""Posterior MEG spectra of the NIMH Healthy Research Volunteer Dataset
(OpenNeuro ds005752, v2.1.0, CC0) and of the empty-room recordings that go
with them.

Why. The adult-aging alpha claim is re-tested in MEG because MEG, unlike
scalp EEG, comes with a measured recording floor: the empty-room recording
(task-noise, the dewar with nobody in it, 100 s at 4800 Hz, recorded about
monthly and shared by the participants of that month). Its spectrum is the
sensor and environmental noise that every fitted background of a resting
spectrum contains (coupling-exponent skill, "The recording floor").

What. For every participant with an age, a ses-01 eyes-closed rest run
(6 min, 1200 Hz, CTF-275) and a ses-01 empty-room recording, the posterior
axial gradiometers (names starting MLO, MRO, MZO, MLP, MRP, MZP) are read
from the rest run and from the empty-room run, without downloading either:
the .res4 header is parsed for channel names, types, gains, sampling rate,
samples per trial and compensation coefficients, and only the bytes of the
needed channels are fetched from the .meg4 file by HTTP range requests (an
8-byte header, then per trial each channel's samples contiguous, int32 big
endian, as MNE's CTF reader assumes). Samples are scaled to fT with
1 / (proper_gain * qgain). The data of this dataset are stored with
third-order gradient compensation (G3BR) applied; a file whose posterior
channels are stored at grade 0 has its reference channels read too and
G3BR applied here, so rest and empty room always enter at grade 3. Trailing
samples that are exactly zero in every channel read (a recording stopped
before the end of its last trial) are dropped. Empty-room recordings longer
than MAX_NOISE_SEC are read for their first MAX_NOISE_SEC only, so that the
floor is estimated from the same duration for everybody.

Spectra as psd_utils: each channel decimated to 300 Hz (polyphase FIR) and
band-passed 0.4-100 Hz, 4-s Hann segments with 50% overlap, segments with
power above 50 times the recording median at any channel and frequency
rejected (psd_utils.reject), fT^2/Hz, 1-100 Hz kept. Stored per recording:
the per-channel mean spectra over the kept segments, and the same over the
kept segments of the first and second half of the recording (segments
straddling the midpoint left out, so the two halves share no data: the
split-floor calibration of meg_nimh_age_alpha.py needs two independent floor
estimates), odd and even kept segments, all segments (no rejection: in rest
recordings the 50x rule also fires on single-channel alpha bursts, at 60 Hz
and above 120 Hz; in empty-room recordings on drift below 1 Hz), and the
ROI-mean spectrum of every segment with its rejection statistic, so other
splits can be formed later.

Output (results/meg_nimh_psd/, not versioned):
  rest_<subject>.npz      one per participant
  noise_<etag>.npz        one per unique empty-room recording (S3 ETag)
  files.csv               participant, age, sex, rest and empty-room keys,
                          sizes, ETag of the empty room, outcome
  transfer.csv            bytes fetched per recording (headers and data)
  headers/                cached .res4 and .hc header files (small)
No raw data are written to disk.

Usage: python meg_nimh_extract_psd.py [--workers 4] [--limit N] [--subjects A,B]
       python meg_nimh_extract_psd.py --check-mne   (reader against MNE, see check_mne)
"""
import argparse
import csv
import io
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from psd_utils import FRANGE, OVERLAP, WIN_SEC, reject, welch_segments  # noqa: E402

S3 = "https://s3.amazonaws.com/openneuro.org"
DS = "ds005752"                  # latest snapshot 2.1.0 = the current S3 objects
OUT = os.path.join(HERE, "..", "results", "meg_nimh_psd")
HDR = os.path.join(OUT, "headers")
ROI_PREFIX = ("MLO", "MRO", "MZO", "MLP", "MRP", "MZP")
SRATE_TARGET = 300.0
MAX_NOISE_SEC = 100.0
HEADER_SIZE = 8                  # "MEG41CP\0" at the start of a .meg4 file
FUNNY_POS = 1844                 # start of the run description in a .res4
G3BR = 0x47334252                # 'G3BR', third-order gradiometer compensation
MEG_CH, REF_MAG, REF_GRAD = 5, 0, 1
CHUNK = 16 << 20                 # bytes per range request
GROUP = 64 << 20                 # bytes of channel data held in memory at once
TRIES = 7                        # attempts per request, waiting 2, 4, ... 64 s between

# .res4 records (big endian, packed), as in MNE's CTF reader (mne/io/ctf/res4.py)
_COIL = np.dtype([("pos", ">f8", 3), ("d0", ">f8"), ("norm", ">f8", 3), ("d1", ">f8"),
                  ("turns", ">i2"), ("d2", ">i4"), ("d3", ">i2"), ("area", ">f8")])
_CH = np.dtype([("sensor_type_index", ">i2"), ("original_run_no", ">i2"), ("coil_type", ">i4"),
                ("proper_gain", ">f8"), ("qgain", ">f8"), ("io_gain", ">f8"),
                ("io_offset", ">f8"), ("num_coils", ">i2"), ("grad_order_no", ">i2"),
                ("d0", ">i4"), ("coil", _COIL, 8), ("head_coil", _COIL, 8)])
_COMP = np.dtype([("sensor_name", "S32"), ("coeff_type", ">i4"), ("d0", ">i4"),
                  ("ncoeff", ">i2"), ("sensors", "S31", 50), ("coeffs", ">f8", 50)])


class Transient(IOError):
    """A request that kept failing for a reason other than a missing file."""


class Transfer:
    """Bytes received, per kind (header, data)."""

    def __init__(self):
        self.bytes = dict(header=0, data=0)
        self.requests = 0


def _retry(what, url, tries=TRIES):
    """Call what(); None if the file does not exist (HTTP 404). Throttling
    (429) and server errors (5xx) and dropped connections are retried with
    exponential backoff and jitter; any other HTTP error is raised at once."""
    err = None
    for k in range(tries):
        try:
            return what()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code != 429 and e.code < 500:
                raise
            err = e
        except Exception as e:
            err = e
        if k < tries - 1:
            time.sleep(2 ** (k + 1) * (0.75 + 0.5 * random.random()))
    raise Transient(f"{type(err).__name__}: {err} ({url})")


def get_bytes(url, tr, kind="header", start=None, stop=None, size=None):
    """Bytes [start, stop) of url (all of it if start is None), retried.
    size: expected total size of the object, checked against Content-Range."""
    hdr = {} if start is None else {"Range": f"bytes={start}-{stop - 1}"}

    def get():
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r:
            if start is not None:
                if r.status != 206:
                    raise IOError(f"range request answered with {r.status}")
                total = int(r.headers["Content-Range"].split("/")[-1])
                if size is not None and total != size:
                    raise ValueError(f"object is {total} bytes, expected {size}")
            want = r.headers.get("Content-Length")
            buf = r.read()
        if want is not None and len(buf) != int(want):
            raise IOError(f"incomplete read, {len(buf)} of {want} bytes")
        if start is not None and len(buf) != stop - start:
            raise IOError(f"range read returned {len(buf)} of {stop - start} bytes")
        return buf
    buf = _retry(get, url)
    if buf is not None:
        tr.bytes[kind] += len(buf)
        tr.requests += 1
    return buf


def s3_list(prefix):
    """[(key, size, etag)] of the S3 objects under prefix."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        url = f"{S3}?{urllib.parse.urlencode(q)}"
        txt = _retry(lambda: urllib.request.urlopen(url, timeout=120).read(), url)
        root = ET.fromstring(txt)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        for c in root.findall(f"{ns}Contents"):
            out.append((c.find(f"{ns}Key").text, int(c.find(f"{ns}Size").text),
                        c.find(f"{ns}ETag").text.strip('"')))
        if root.findtext(f"{ns}IsTruncated") != "true":
            return out
        token = root.findtext(f"{ns}NextContinuationToken")


# ---------------------------------------------------------------- headers

def parse_res4(buf):
    """Fields of a CTF .res4 header needed to read and scale channels.

    Offsets as in MNE's _read_res4 (fixed fields up to the run-description
    length at 1836, the run description at 1844, then the filters, the
    channel names (32 bytes each), the channel records and the compensation
    records). Coil positions are in cm, dewar coordinates.
    """
    def i2(o):
        return int(np.frombuffer(buf, ">i2", 1, o)[0])

    def i4(o):
        return int(np.frombuffer(buf, ">i4", 1, o)[0])

    def text(o, n):
        return buf[o:o + n].split(b"\0")[0].decode("latin-1")

    r = dict(data_time=text(778, 255), data_date=text(1033, 255), nsamp=i4(1288),
             nchan=i2(1292), sfreq=float(np.frombuffer(buf, ">f8", 1, 1296)[0]),
             no_trials=i2(1312))
    if "/" in r["data_time"] and ":" in r["data_date"]:    # swapped in some files
        r["data_time"], r["data_date"] = r["data_date"], r["data_time"]
    pos = FUNNY_POS + i4(1836)
    nfilt = i2(pos)
    pos += 2
    for _ in range(nfilt):
        pos += 16
        pos += 2 + 8 * i2(pos)
    n = r["nchan"]
    r["ch_names"] = [text(pos + 32 * k, 32) for k in range(n)]
    pos += 32 * n
    chs = np.frombuffer(buf, _CH, n, pos)
    pos += n * _CH.itemsize
    r["type"] = chs["sensor_type_index"].astype(int)
    r["grade"] = chs["grad_order_no"].astype(int)
    r["gain"] = (chs["proper_gain"] * chs["qgain"]).astype(float)   # counts per tesla
    r["pos"] = chs["coil"]["pos"][:, 0, :].astype(float)            # first coil, cm
    ncomp = i2(pos)
    comps = np.frombuffer(buf, _COMP, ncomp, pos + 2)
    r["comp"] = {}
    for c in comps:
        if int(c["coeff_type"]) != G3BR:
            continue
        k = int(c["ncoeff"])
        r["comp"][c["sensor_name"].split(b"\0")[0].decode("latin-1")] = (
            [s.split(b"\0")[0].decode("latin-1") for s in c["sensors"][:k]],
            c["coeffs"][:k].astype(float))
    return r


def parse_hc(txt):
    """Measured nasion, left and right ear coil positions relative to the
    dewar (cm) from a CTF .hc file; None if absent."""
    out = {}
    for name, key in (("nasion", "nas"), ("left ear", "lpa"), ("right ear", "rpa")):
        m = re.search(rf"measured {name} coil position relative to dewar \(cm\):\s*"
                      r"x = (\S+)\s*y = (\S+)\s*z = (\S+)", txt)
        if not m:
            return None
        out[key] = np.array([float(v) for v in m.groups()])
    return out


def header(url, local, tr):
    """The bytes of a small header file, from the cache or fetched (and cached)."""
    if os.path.exists(local):
        with open(local, "rb") as fh:
            return fh.read()
    buf = get_bytes(url, tr, "header")
    if buf is None:
        return None
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local + ".part", "wb") as fh:
        fh.write(buf)
    os.replace(local + ".part", local)
    return buf


# ---------------------------------------------------------------- data

def roi_index(res4):
    """Indices of the posterior axial gradiometers."""
    return [k for k, (c, t) in enumerate(zip(res4["ch_names"], res4["type"]))
            if t == MEG_CH and c.startswith(ROI_PREFIX)]


def byte_ranges(res4, idx, n_trial, n_use):
    """(start, stop, channel, trial) of each channel block to read; contiguous
    blocks are merged later. n_use: samples per trial to read (<= nsamp)."""
    ns, nc = res4["nsamp"], res4["nchan"]
    out = []
    for t in range(n_trial):
        for k in idx:
            s = HEADER_SIZE + (t * nc + k) * ns * 4
            out.append((s, s + n_use * 4, k, t))
    return out


def read_channels(fetch, res4, idx, size, max_sec=None, dec=None):
    """Read channels idx (sorted) of a .meg4 file: fT, channels x samples.

    fetch(start, stop) returns bytes. Channel blocks that follow one another
    in the file are fetched together, in requests of at most CHUNK bytes,
    holding at most GROUP bytes at a time. dec: decimation factor, applied
    per group (polyphase FIR) to keep memory low. Samples after the first
    sample from which every channel read is exactly 0 to the end (a
    recording stopped early; MNE finds the same point from the system clock
    channel) are dropped before decimation.
    """
    ns, nc = res4["nsamp"], res4["nchan"]
    per_trial = 4 * nc * ns
    if (size - HEADER_SIZE) % per_trial:
        raise ValueError(f"file of {size} bytes is not a whole number of trials "
                         f"({nc} channels x {ns} samples): truncated")
    n_trial = (size - HEADER_SIZE) // per_trial
    n_use, n_trial_use = ns, n_trial
    if max_sec is not None:
        want = int(round(max_sec * res4["sfreq"]))
        if want < ns:
            n_use, n_trial_use = want, 1
        else:
            n_trial_use = min(n_trial, int(np.ceil(want / ns)))
    rng = byte_ranges(res4, idx, n_trial_use, n_use)
    # group runs of adjacent blocks of one trial, at most GROUP bytes each
    groups, cur = [], []
    for r in rng:
        if cur and (r[0] != cur[-1][1] or (r[1] - cur[0][0]) > GROUP or r[3] != cur[-1][3]):
            groups.append(cur)
            cur = []
        cur.append(r)
    groups.append(cur)
    pos = {k: j for j, k in enumerate(idx)}
    gain = res4["gain"]
    parts, valid = {}, np.zeros(n_trial_use, int)
    for g in groups:
        a, b = g[0][0], g[-1][1]
        buf = np.empty(b - a, np.uint8)
        for s in range(a, b, CHUNK):
            e = min(s + CHUNK, b)
            buf[s - a:e - a] = np.frombuffer(fetch(s, e), np.uint8)
        xi = buf.view(">i4").reshape(len(g), n_use)
        t = g[0][3]
        nz = np.flatnonzero(np.any(xi != 0, axis=0))
        valid[t] = max(valid[t], nz[-1] + 1 if nz.size else 0)
        ks = [r[2] for r in g]
        x = xi.astype(np.float64) * (1e15 / gain[ks])[:, None]
        del buf, xi
        if dec and dec > 1:
            x = resample_poly(x, 1, dec, axis=1)
        for k, row in zip(ks, x):
            parts.setdefault(t, {})[pos[k]] = row
    short = np.flatnonzero(valid < n_use)
    n_valid = (short[0] * n_use + valid[short[0]]) if short.size else n_trial_use * n_use
    x = np.vstack([np.concatenate([parts[t][j] for t in sorted(parts)])
                   for j in range(len(idx))])
    return x[:, :int(n_valid // (dec or 1))]


def compensate(res4, data, names, refs, ref_names):
    """Apply G3BR to channels stored at grade 0 (fT in, fT out).

    MNE's convention (mne/io/ctf/info.py _conv_comp): the coefficient c_ij of
    reference j for channel i is scaled by gain_j / gain_i, then
    x_i <- x_i - sum_j c'_ij r_j in physical units.
    """
    g = dict(zip(res4["ch_names"], res4["gain"]))
    out = data.copy()
    for i, nm in enumerate(names):
        sens, coef = res4["comp"][nm]
        for s, c in zip(sens, coef):
            if c != 0.0:
                out[i] -= c * g[s] / g[nm] * refs[ref_names.index(s)]
    return out


def preprocess(x, srate):
    """0.4-100 Hz band-pass at the decimated rate, as psd_utils.preprocess."""
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(x, srate, 0.4, min(srate / 2 - 5, 100.0),
                                      fir_design="firwin", verbose="ERROR")


def spectra(x, srate):
    """Welch spectra of channels x samples (fT) and their summaries (see top)."""
    P, f, _ = welch_segments(x, srate)
    nper = int(round(WIN_SEC * srate))
    step = nper - int(round(OVERLAP * nper))
    keep = reject(P)                                   # over all channels and 0-150 Hz
    ratio = np.max(P / np.maximum(np.median(P, axis=0, keepdims=True), 1e-12), axis=(1, 2))
    fm = (f >= FRANGE[0]) & (f <= FRANGE[1])
    P = P[:, :, fm]
    start = np.arange(P.shape[0]) * step
    mid = x.shape[1] / 2.0
    h1 = keep & (start + nper <= mid)
    h2 = keep & (start >= mid)
    ki = np.flatnonzero(keep)
    odd, even = np.zeros_like(keep), np.zeros_like(keep)
    odd[ki[0::2]], even[ki[1::2]] = True, True

    def mean(m):
        return P[m].mean(0) if m.any() else np.full(P.shape[1:], np.nan, np.float32)
    return dict(f=f[fm], P=mean(keep), P_h1=mean(h1), P_h2=mean(h2), P_odd=mean(odd),
                P_even=mean(even), P_all=mean(np.ones_like(keep)),
                seg=P.mean(1).astype(np.float32), seg_start_s=start / srate,
                seg_keep=keep, seg_ratio=ratio.astype(np.float32),
                n_seg=int(keep.sum()), n_seg_total=int(keep.size),
                n_h1=int(h1.sum()), n_h2=int(h2.sum()), dur_s=x.shape[1] / srate)


def recording(url, size, res4, tr, max_sec=None):
    """ROI data (fT, 300 Hz, grade 3) of one .meg4, and its channel names."""
    idx = roi_index(res4)
    names = [res4["ch_names"][k] for k in idx]
    dec = int(round(res4["sfreq"] / SRATE_TARGET))
    def fetch(a, b):
        buf = get_bytes(url, tr, "data", a, b, size)
        if buf is None:
            raise Transient(f"{url} not found")
        return buf
    grades = set(res4["grade"][idx])
    if grades == {3}:
        x = read_channels(fetch, res4, idx, size, max_sec, dec)
        grade_stored = 3
    elif grades == {0}:
        # read the references too, compensate at the full rate, then decimate
        ref_idx = sorted({res4["ch_names"].index(s) for nm in names
                          for s, c in zip(*res4["comp"][nm]) if c != 0.0})
        ref_names = [res4["ch_names"][k] for k in ref_idx]
        both = sorted(set(idx) | set(ref_idx))
        raw = read_channels(fetch, res4, both, size, max_sec, None)
        sel = [both.index(k) for k in idx]
        rsel = [both.index(k) for k in ref_idx]
        x = compensate(res4, raw[sel], names, raw[rsel], ref_names)
        x = resample_poly(x, 1, dec, axis=1)[:, :raw.shape[1] // dec]
        grade_stored = 0
    else:
        raise ValueError(f"posterior channels stored at mixed grades {sorted(grades)}")
    return x, names, res4["sfreq"] / dec, grade_stored, dec


# ---------------------------------------------------------------- jobs

def ds_url(key):
    return f"{S3}/{key}"


def job(kind, name, key, size, etag):
    """Spectra of one rest (kind 'rest', name = subject) or empty-room (kind
    'noise', name = ETag) recording. Returns a transfer row."""
    tr = Transfer()
    dsdir = os.path.dirname(key)
    base = os.path.basename(dsdir)[:-3]
    out = os.path.join(OUT, f"{kind}_{name}.npz")
    t0 = time.time()
    res4 = parse_res4(header(ds_url(f"{dsdir}/{base}.res4"),
                             os.path.join(HDR, base, base + ".res4"), tr))
    extra = {}
    if kind == "rest":
        hc = header(ds_url(f"{dsdir}/{base}.hc"), os.path.join(HDR, base, base + ".hc"), tr)
        hc = parse_hc(hc.decode("latin-1")) if hc else None
        if hc:
            extra.update(hc_nas=hc["nas"], hc_lpa=hc["lpa"], hc_rpa=hc["rpa"])
    x, names, srate, grade0, dec = recording(
        ds_url(key), size, res4, tr, MAX_NOISE_SEC if kind == "noise" else None)
    x = preprocess(x, srate)
    s = spectra(x, srate)
    idx = [res4["ch_names"].index(c) for c in names]
    np.savez_compressed(out + ".part.npz", kind=kind, name=name, key=key, size=size, etag=etag,
                        ch_names=np.array(names), sensor_pos_cm=res4["pos"][idx],
                        sfreq_orig=res4["sfreq"], srate=srate, dec=dec,
                        nsamp=res4["nsamp"], no_trials=res4["no_trials"],
                        grade_stored=grade0, data_date=res4["data_date"],
                        data_time=res4["data_time"], **extra, **s)
    os.replace(out + ".part.npz", out)
    return dict(kind=kind, name=name, key=key, size=size, n_ch=len(names),
                header_bytes=tr.bytes["header"], data_bytes=tr.bytes["data"],
                requests=tr.requests, seconds=round(time.time() - t0, 1),
                dur_s=round(s["dur_s"], 2), n_seg=s["n_seg"], n_seg_total=s["n_seg_total"],
                grade_stored=grade0, status="done")


def file_table(subjects=None):
    """Participants with age, sex and their ses-01 rest run-01 and empty-room
    .meg4 (key, size, ETag), from participants.tsv and the S3 listing."""
    tr = Transfer()
    txt = header(f"{S3}/{DS}/participants.tsv", os.path.join(OUT, "participants.tsv"), tr)
    rows = list(csv.DictReader(io.StringIO(txt.decode("utf-8")), delimiter="\t"))
    out = []
    for r in rows:
        sub = r["participant_id"]
        if r.get("MEG") != "1" or (subjects and sub not in subjects):
            continue
        objs = s3_list(f"{DS}/{sub}/ses-01/meg/")
        def find(task):
            m = [o for o in objs if o[0].endswith(f"_task-{task}_run-01_meg.meg4")]
            return m[0] if m else (None, None, None)
        rk, rs, _ = find("rest")
        nk, ns, ne = find("noise")
        try:
            age = float(r["age"])
        except ValueError:
            age = float("nan")
        out.append(dict(subject=sub, age=age, sex=r["sex"], rest_key=rk, rest_size=rs,
                        noise_key=nk, noise_size=ns, noise_etag=ne))
    return out


def check_mne():
    """Check the reader against mne.io.read_raw_ctf on one small file.

    The file (sub-ON02747 task-artifact, 303 channels x 31200 samples, 38 MB)
    is streamed into memory; MNE reads the same bytes through a patched open()
    (headers from the local cache, an empty placeholder for the .meg4 path),
    so no raw data touch the disk. Compared:
    the posterior channels at the stored grade (3), our G3BR applied to MNE's
    grade-0 data against the stored grade-3 data, and a range-read of two
    channels over HTTP against the in-memory bytes.
    """
    import builtins
    import mne
    import mne.io.ctf.ctf as mctf

    tr = Transfer()
    base = "sub-ON02747_ses-01_task-artifact_run-01_meg"
    dsdir = f"{DS}/sub-ON02747/ses-01/meg/{base}.ds"
    local = os.path.join(HDR, "mne_check", base + ".ds")
    for ext in ("res4", "hc", "infods"):
        header(ds_url(f"{dsdir}/{base}.{ext}"), os.path.join(local, f"{base}.{ext}"), tr)
    blob = get_bytes(ds_url(f"{dsdir}/{base}.meg4"), tr, "data")
    print(f"streamed {len(blob) / 1e6:.1f} MB into memory")
    meg4 = os.path.join(local, base + ".meg4")

    class Mem(io.BytesIO):
        pass

    real_open, real_np = builtins.open, mctf.np

    def fake_open(fn, mode="r", *a, **k):
        if os.path.abspath(str(fn)) == os.path.abspath(meg4):
            return Mem(blob)
        return real_open(fn, mode, *a, **k)

    class NP:
        def __getattr__(self, k):
            return getattr(real_np, k)

        def fromfile(self, fid, dtype, count=-1, *a, **k):
            if isinstance(fid, Mem):
                dt = np.dtype(dtype)
                n = dt.itemsize * count if count >= 0 else -1
                return np.frombuffer(fid.read(n), dt).copy()
            return real_np.fromfile(fid, dtype, count, *a, **k)

    real_mk = mctf._make_ctf_name

    def fake_mk(directory, extra, raise_error=True):
        fn, found = real_mk(directory, extra, raise_error=False)
        if extra == "meg4":
            return fn, True
        if not found and raise_error:
            raise OSError(fn)
        return fn, found

    mctf.open, mctf.np, mctf._make_ctf_name = fake_open, NP(), fake_mk
    real_open(meg4, "wb").close()          # empty placeholder: MNE checks that the path exists
    try:
        raw = mne.io.read_raw_ctf(local, preload=True, verbose="ERROR")
    finally:
        del mctf.open
        mctf.np, mctf._make_ctf_name = real_np, real_mk
        os.remove(meg4)
    with open(os.path.join(local, base + ".res4"), "rb") as fh:
        res4 = parse_res4(fh.read())
    from mne.io.ctf.res4 import _read_res4
    m4 = _read_res4(local)
    assert m4["ch_names"] == res4["ch_names"] and m4["nsamp"] == res4["nsamp"]
    assert np.allclose([c["proper_gain"] * c["qgain"] for c in m4["chs"]], res4["gain"])
    n_comp = sum(c["coeff_type"] == G3BR for c in m4["comp"])
    assert n_comp == len(res4["comp"])
    idx = roi_index(res4)
    names = [res4["ch_names"][k] for k in idx]
    fetch = lambda a, b: blob[a:b]                                        # noqa: E731
    ours = read_channels(fetch, res4, idx, len(blob))
    theirs = raw.get_data(picks=names) * 1e15
    print(f"ROI {len(idx)} channels; stored grade {raw.compensation_grade}; "
          f"max |ours - MNE| {np.max(np.abs(ours - theirs)):.3g} fT "
          f"(signal s.d. {np.std(theirs):.3g} fT)")
    assert np.allclose(ours, theirs, rtol=1e-10, atol=1e-6)
    # compensation: MNE's grade 0, our G3BR, against the stored grade 3
    raw0 = raw.copy().apply_gradient_compensation(0, verbose="ERROR")
    ref_names = [c for c, t in zip(res4["ch_names"], res4["type"]) if t in (REF_MAG, REF_GRAD)]
    x0 = raw0.get_data(picks=names) * 1e15
    r0 = raw0.get_data(picks=ref_names) * 1e15
    x3 = compensate(res4, x0, names, r0, ref_names)
    print(f"G3BR from grade 0: max |ours - stored| {np.max(np.abs(x3 - theirs)):.3g} fT; "
          f"grade 0 differs from 3 by {np.max(np.abs(x0 - theirs)):.3g} fT")
    assert np.allclose(x3, theirs, rtol=1e-6, atol=1e-3)
    # the HTTP range path for two channels
    k = idx[:2]
    url = ds_url(f"{dsdir}/{base}.meg4")
    http = read_channels(lambda a, b: get_bytes(url, tr, "data", a, b, len(blob)),
                         res4, k, len(blob))
    assert np.array_equal(http, ours[:2])
    print(f"HTTP range read of 2 channels identical; "
          f"transferred {sum(tr.bytes.values()) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="concurrent recordings (connections)")
    ap.add_argument("--limit", type=int, default=0, help="first N participants only")
    ap.add_argument("--subjects", default="", help="comma-separated participant IDs")
    ap.add_argument("--check-mne", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.check_mne:
        check_mne()
        return
    subs = set(a.subjects.split(",")) if a.subjects else None
    fpath = os.path.join(OUT, "files.csv")
    if os.path.exists(fpath) and not subs:
        with open(fpath, encoding="utf-8") as fh:
            files = list(csv.DictReader(fh))
        for r in files:
            for c in ("age", "rest_size", "noise_size"):
                r[c] = float(r[c]) if r[c] not in ("", "None", "nan") else None
    else:
        files = file_table(subs)
        if not subs:
            with open(fpath, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(files[0]))
                w.writeheader()
                w.writerows(files)
    usable = [r for r in files if r["rest_key"] and r["noise_key"] and r["age"] is not None
              and np.isfinite(r["age"])]
    if a.limit:
        usable = usable[:a.limit]
    print(f"{len(files)} participants with MEG; {len(usable)} with age, rest and empty room")
    jobs, seen = [], set()
    for r in usable:
        if r["noise_etag"] not in seen:
            seen.add(r["noise_etag"])
            jobs.append(("noise", r["noise_etag"], r["noise_key"], int(r["noise_size"]),
                         r["noise_etag"]))
        jobs.append(("rest", r["subject"], r["rest_key"], int(r["rest_size"]), ""))
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}_{j[1]}.npz"))]
    print(f"{len(jobs)} recordings ({len(seen)} unique empty rooms), {len(todo)} to do")
    tpath = os.path.join(OUT, "transfer.csv")
    fields = ["kind", "name", "key", "size", "n_ch", "header_bytes", "data_bytes", "requests",
              "seconds", "dur_s", "n_seg", "n_seg_total", "grade_stored", "status"]
    new = not os.path.exists(tpath)
    with open(tpath, "a", newline="", encoding="utf-8") as fh, \
            ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        futs = {ex.submit(job, *j): j for j in todo}
        tot = 0
        for k, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            try:
                row = fu.result()
            except Exception as e:
                row = dict(kind=j[0], name=j[1], key=j[2], size=j[3],
                           status=f"failed: {type(e).__name__}: {str(e)[:200]}")
            w.writerow(row)
            fh.flush()
            tot += row.get("data_bytes", 0) + row.get("header_bytes", 0)
            print(f"[{k}/{len(todo)}] {j[0]} {j[1]} {row['status'][:80]} "
                  f"{row.get('data_bytes', 0) / 1e6:.0f} MB in {row.get('seconds', 0)} s; "
                  f"total {tot / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
