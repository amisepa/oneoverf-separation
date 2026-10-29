"""IRASA on the Welch segments of continuous blocks (Wen & Liu, 2016).

The time-domain algorithm of code/lib/oof_est_irasa.m: every block is
resampled by h and by 1/h (polyphase, scipy's resample_poly), the Welch
segments are cut from the resampled blocks at the matching positions, and
the periodograms of the segments in use are averaged per factor. The fractal
spectrum is the median over h of the geometric mean of the up- and
down-sampled averages. Averaging comes before the geometric mean and the
median, as in Wen & Liu: the median of single-segment periodograms (close to
exponentially distributed) sits about 30% below their mean.

A rhythm at f0 lands at f0/h and f0*h in the two resampled spectra and the
median over h removes most of it; a power law is not displaced, since the
geometric mean of h^(1-chi) and h^(chi-1) is 1. The estimate at f reads the
original spectrum at f*h and f/h, so it is valid only where both lie inside
the data's pass band for every h (HBN extraction, 0.4-100 Hz, hmax 1.9:
about 0.8-52 Hz). The oscillatory spectrum is the Welch spectrum of the same
segments minus the fractal one.
"""
from fractions import Fraction

import numpy as np
from scipy.fft import next_fast_len, rfft
from scipy.signal import resample_poly

HSET = np.round(np.arange(1.1, 1.9001, 0.05), 2)   # Wen & Liu's default, as oof_est_irasa


def _psd_sum(y, srate, starts, m, nfft):
    """Sum over segments of the one-sided Hann periodogram (uV^2/Hz) of the
    length-m segments of y (nch, nsamp) starting at starts; (nch, nfft//2+1),
    on rfftfreq(nfft, 1/srate). Same window and scaling as
    hbn_extract_psd.welch; zero-padding to nfft only refines the grid.
    Segments and FFT in float32, sums in float64: 1.8x faster, 3e-7 from
    float64."""
    seg = y[:, starts[:, None] + np.arange(m)].astype(np.float32)   # (nch, nseg, m)
    seg -= seg.mean(axis=2, keepdims=True)
    win = np.hanning(m + 1)[:m]
    seg *= win.astype(np.float32)
    Z = rfft(seg, n=nfft, axis=2)
    p = (Z.real ** 2 + Z.imag ** 2).sum(axis=1, dtype=np.float64)
    p *= 2.0 / (srate * np.sum(win ** 2))
    p[:, 0] /= 2.0
    if nfft % 2 == 0:
        p[:, -1] /= 2.0
    return p


def _interp(p, fm, f):
    """Linear interpolation of p (nch, len(fm)) onto f, all channels at once."""
    j = np.clip(np.searchsorted(fm, f), 1, fm.size - 1)
    w = (f - fm[j - 1]) / (fm[j] - fm[j - 1])
    return p[:, j - 1] * (1 - w) + p[:, j] * w


def irasa(X, srate, spans, nper, noverlap, used, f, hset=HSET):
    """X: (nch, nsamp); spans: (start, stop) samples of the blocks; nper,
    noverlap: the Welch segments of hbn_extract_psd.welch (spans shorter than
    nper have none); used: bool per segment in time order, as blocks_psd's
    seg_used; f: output frequencies (Hz).

    Returns frac (nch, nf), the fractal spectrum, and hspec (nh, 2, nch, nf),
    the mean up- (index 0, resampled by h) and down-sampled (1, by 1/h)
    periodograms per factor, from which frac (or the fractal spectrum of a
    channel average) can be formed again."""
    step = nper - noverlap
    used = np.asarray(used, bool)
    blocks, k = [], 0
    for a, b in spans:
        if b - a < nper:
            continue
        n = 1 + (b - a - nper) // step
        u = used[k:k + n]
        k += n
        if u.any():
            blocks.append((a, b, (np.arange(n) * step)[u]))
    if k != used.size:
        raise ValueError(f"{used.size} segment flags for {k} segments")
    nseg = int(used.sum())
    hspec = np.empty((len(hset), 2, X.shape[0], len(f)))
    for i, h in enumerate(hset):
        r = Fraction(float(h)).limit_denominator(100)
        for j, (up, down) in enumerate(((r.numerator, r.denominator),
                                        (r.denominator, r.numerator))):
            m = int(round(nper * up / down))
            nfft = next_fast_len(m, real=True)
            acc = np.zeros((X.shape[0], nfft // 2 + 1))
            for a, b, rel in blocks:
                y = resample_poly(X[:, a:b], up, down, axis=1)
                starts = np.minimum(np.round(rel * up / down).astype(int), y.shape[1] - m)
                acc += _psd_sum(y, srate, starts, m, nfft)
            hspec[i, j] = _interp(acc / nseg, np.fft.rfftfreq(nfft, 1.0 / srate), f)
    frac = np.median(np.sqrt(hspec[:, 0] * hspec[:, 1]), axis=0)
    return frac, hspec
