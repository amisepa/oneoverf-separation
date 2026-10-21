"""Population spiking, channel-mean LFP and short-window spectra of the
Trujillo et al. (2019) organoid recordings, for an event-locked analysis.

Per recording (Zenodo 4751759, one file per week; see trujillo_extract_psd.py
for the file formats) and well:

    pop<k>   population spiking, as in the paper: multi-unit spike times of all
             64 channels binned at 1 ms, summed, smoothed with a normalised
             100-point Gaussian window (MATLAB gausswin, alpha 2.5), in
             spikes/s; stored every 10 ms
    lfp<k>   mean LFP over the channels kept (standard deviation within 0.2-5 x
             the well median), decimated to 250 Hz (V)
    S<k>     spectra of 3-s windows starting every 0.25 s (Hann, zero-padded
             to 4 s: 0.25 Hz grid, 0-100 Hz), the mean over the channels kept
             of each channel's spectrum (V^2/Hz); window starts in win<k>
             (samples at 1 kHz)
    good<k>  channels kept

Event and quiet windows are chosen afterwards (trujillo_events.py), so the
definitions can change without streaming the data again. Spike times are
read from the 'spikes' variable (sample numbers at 12.5 kHz; both clocks
start at 0).

Writes OUT/YYMMDD.npz.

Usage: python trujillo_extract_events.py LFP_Sp_161216.mat [--out DIR] [--wells 4,...,11]
"""
import argparse
import os
import time
import urllib.request

import fsspec
import h5py
import numpy as np
from scipy.signal import decimate

from trujillo_extract_psd import URL, cells_v5

FS = 1000.0
FS_SPIKE = 12500.0
WIN, NFFT, STEP = 3000, 4000, 250
FMAX = 100.0


def population_spiking(trains, n):
    pop = np.zeros(n)
    for t in trains:
        idx = np.floor(np.asarray(t, float).ravel() / FS_SPIKE * FS).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        np.add.at(pop, idx, 1.0)
    alpha = 2.5
    m = np.arange(100) - 49.5
    g = np.exp(-0.5 * (alpha * m / 49.5) ** 2)
    g /= g.sum()
    return np.convolve(pop, g, mode="same") * FS


def window_spectra(x, good):
    """Mean over good channels of Hann-window spectra, every STEP samples."""
    n = x.shape[1]
    starts = np.arange(0, n - WIN + 1, STEP)
    w = np.hanning(WIN + 1)[:WIN].astype(np.float32)
    scale = 2.0 / (FS * np.sum(w.astype(float) ** 2))
    f = np.fft.rfftfreq(NFFT, 1.0 / FS)
    keep_f = f <= FMAX
    xg = x[good].astype(np.float32)
    out = np.empty((starts.size, keep_f.sum()), np.float32)
    for i0 in range(0, starts.size, 64):
        st = starts[i0:i0 + 64]
        seg = np.stack([xg[:, s:s + WIN] for s in st], 1)        # chan x win x time
        seg -= seg.mean(-1, keepdims=True)
        p = np.abs(np.fft.rfft(seg * w, n=NFFT, axis=-1)) ** 2 * scale
        out[i0:i0 + len(st)] = p[..., keep_f].mean(0)
    return starts, f[keep_f], out


def good_channels(x):
    rms = x.std(1)
    med = np.median(rms[rms > 0])
    return (rms > 0.2 * med) & (rms < 5 * med)


def process(w, x, trains, res):
    n = x.shape[1]
    good = good_channels(x)
    pop = population_spiking(trains, n)
    starts, f, S = window_spectra(x, good)
    res[f"pop{w}"] = pop[::10].astype(np.float32)
    res[f"lfp{w}"] = decimate(x[good].mean(0), 4, zero_phase=True).astype(np.float32)
    res[f"S{w}"] = S
    res[f"win{w}"] = starts
    res[f"good{w}"] = good
    res[f"n{w}"] = n
    res[f"nspk{w}"] = int(sum(np.asarray(t).size for t in trains))
    return f


def h5_cell(h, ref):
    """One cell of a MATLAB 7.3 cell array. An empty array is stored as its
    dimensions (e.g. [0, 0]) with the attribute MATLAB_empty; reading it as
    data would give two spikes at sample 0."""
    d = h[ref]
    if d.attrs.get("MATLAB_empty"):
        return np.zeros(0)
    return d[()]


def repair_spikes(fn, path, wells):
    """Recompute pop<k> and nspk<k> of an existing output from the spike
    times only (for files written before empty cells were handled)."""
    d = dict(np.load(path))
    url = URL.format(fn)
    with fsspec.filesystem("http").open(url, "rb", block_size=2 ** 22,
                                        cache_type="readahead") as fh, h5py.File(fh, "r") as h:
        srefs = h["spikes"][()]
        for w in wells:
            trains = [h5_cell(h, srefs[c, w]) for c in range(64)]
            n = int(d[f"n{w}"])
            d[f"pop{w}"] = population_spiking(trains, n)[::10].astype(np.float32)
            d[f"nspk{w}"] = int(sum(np.asarray(t).size for t in trains))
    np.savez(path, **d)


def extract(fn, out, wells):
    res, f = {}, None
    t0 = time.time()
    url = URL.format(fn)
    head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-127"}),
                                  timeout=60).read()
    if b"MATLAB 7.3" in head:
        with fsspec.filesystem("http").open(url, "rb", block_size=2 ** 23,
                                            cache_type="readahead") as fh, h5py.File(fh, "r") as h:
            srefs, lrefs = h["spikes"][()], h["LFP"][()].ravel()     # spikes: 64 x 12
            for w in wells:
                trains = [h5_cell(h, srefs[c, w]) for c in range(64)]
                f = process(w, h[lrefs[w]][()], trains, res)
                print(fn, "well", w + 1, f"{time.time() - t0:.0f} s", flush=True)
    else:
        spk = {k: [] for k in wells}
        want = {"spikes": {w + 12 * c for w in wells for c in range(64)}, "LFP": set(wells)}
        for name, k, arr in cells_v5(url, want):
            if name == "spikes":
                spk[k % 12].append(arr)
            else:
                if len(spk[k]) != 64:
                    raise ValueError(f"{fn}: LFP of well {k + 1} before its spike times")
                f = process(k, arr, spk[k], res)
                print(fn, "well", k + 1, f"{time.time() - t0:.0f} s", flush=True)
    np.savez(out, f=f, fs=FS, step=STEP, win=WIN, **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "trujillo_events"))
    ap.add_argument("--wells", default="4,5,6,7,8,9,10,11")
    ap.add_argument("--repair-spikes", action="store_true",
                    help="recompute pop and nspk of an existing output (MATLAB 7.3 files)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    date = a.file.replace("LFP_Sp_", "").replace(".mat", "")
    wells = [int(w) for w in a.wells.split(",")]
    if a.repair_spikes:
        repair_spikes(a.file, os.path.join(a.out, date + ".npz"), wells)
    else:
        extract(a.file, os.path.join(a.out, date), wells)


if __name__ == "__main__":
    main()
