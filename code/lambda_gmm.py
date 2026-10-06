"""Within-subject coupling exponent without logging the periodic power.

For two conditions (1, 2) the band-power model a = c b^lambda, with an
intrinsic change delta common to all subjects (c2 = c1 e^delta), gives for
every subject

    r_2 - e^delta r_1 = 0,   r_k = a_k / b_k^lambda,

so lambda and delta solve the moment conditions

    sum_i w_i (r_i2 - e^delta r_i1)        = 0
    sum_i w_i z_i (r_i2 - e^delta r_i1)    = 0

with z_i the centred change in ln b. Unlike a regression of the change in
ln a on the change in ln b, this needs no a > 0: subjects whose periodic
power estimate is negative stay in, so the estimate is not driven by the
selection of positive residuals, which scale with the background and look
like lambda = 1.

Estimation noise. With two halves of the Welch segments (A, B), the band's
total power t comes from half A, the background b used in a = t - b and in
r from half B, and the instrument z and the weights from half A's background
fit, which uses only frequencies outside the band (a censored or flanking
fit). The three are then estimated from disjoint data, so noise in b cannot
attenuate lambda as it does when one estimate of b appears in both r and z.
Both assignments (A, B) = (1, 2) and (2, 1) are pooled.

The instrument and the weights can instead come from half B, the half that
supplies b (instrument="background"); t is then the quantity estimated from
separate data. The two choices agree when the halves share one true
background. When the true background differs between the halves of a
recording, taking the instrument from the half of t reads high and taking it
from the half of b reads low; the gap between the two measures that
sensitivity (hbn_half_assignment.py).

With three disjoint sets of segments (instrument="third", arrays with three
columns) t, b and the instrument each come from a different set, and the six
ways of assigning them are pooled. No two of the three then share a set, so
neither a background that differs between the sets nor noise in its fit
enters the moment twice. The sets are smaller, so the estimate is noisier.
"""
import numpy as np
from scipy import optimize
from scipy.optimize import brentq


def _stack(t1, t2, b1, b2, instrument="total"):
    """Pool the two half assignments into flat arrays.

    instrument: the half whose background fit gives the instrument and the
    weights, "total" (the half of t) or "background" (the half of b);
    "third" takes three replicates and pools the six assignments (_stack3).
    """
    if instrument == "third":
        return _stack3(t1, t2, b1, b2)
    if instrument not in ("total", "background"):
        raise ValueError("instrument must be 'total', 'background' or 'third'")
    T1, T2, B1, B2, Z, W = [], [], [], [], [], []
    for sa, sb in ((0, 1), (1, 0)):
        si = sa if instrument == "total" else sb
        ok = (np.all(np.isfinite([t1[:, sa], t2[:, sa], b1[:, sb], b2[:, sb],
                                  b1[:, sa], b2[:, sa]]), 0)
              & (b1[:, sb] > 0) & (b2[:, sb] > 0) & (b1[:, sa] > 0) & (b2[:, sa] > 0))
        T1.append(t1[ok, sa]); T2.append(t2[ok, sa])
        B1.append(b1[ok, sb]); B2.append(b2[ok, sb])
        z = np.log(b2[ok, si]) - np.log(b1[ok, si])
        Z.append(z - z.mean())
        W.append(np.sqrt(b1[ok, si] * b2[ok, si]))       # scale from the same half
    return [np.concatenate(v) for v in (T1, T2, B1, B2, Z, W)]


def _stack3(t1, t2, b1, b2):
    """Pool the six assignments of three replicates into flat arrays.

    Arrays of shape (n_subjects, 3). In each assignment t comes from
    replicate sa, b from sb, and the instrument and the weights from the
    background fit of si, with sa, sb, si all different.
    """
    if any(np.shape(v)[1] != 3 for v in (t1, t2, b1, b2)):
        raise ValueError("instrument='third' needs three replicates per condition")
    T1, T2, B1, B2, Z, W = [], [], [], [], [], []
    for sa, sb, si in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        ok = (np.all(np.isfinite([t1[:, sa], t2[:, sa], b1[:, sb], b2[:, sb],
                                  b1[:, si], b2[:, si]]), 0)
              & (b1[:, sb] > 0) & (b2[:, sb] > 0) & (b1[:, si] > 0) & (b2[:, si] > 0))
        T1.append(t1[ok, sa]); T2.append(t2[ok, sa])
        B1.append(b1[ok, sb]); B2.append(b2[ok, sb])
        z = np.log(b2[ok, si]) - np.log(b1[ok, si])
        Z.append(z - z.mean())
        W.append(np.sqrt(b1[ok, si] * b2[ok, si]))
    return [np.concatenate(v) for v in (T1, T2, B1, B2, Z, W)]


def _m2(lam, T1, T2, B1, B2, Z, S):
    """Second moment, profiled over delta; weights 1 / S^(1 - lambda)."""
    W = S ** (lam - 1.0)
    R1 = (T1 - B1) / B1 ** lam
    R2 = (T2 - B2) / B2 ** lam
    s1 = np.sum(W * R1)
    if s1 == 0:
        return np.nan, np.nan
    ed = np.sum(W * R2) / s1
    return np.sum(W * Z * (R2 - ed * R1)) / np.sum(W), ed


def estimate(t1, t2, b1, b2, grid=None, near=None, instrument="total"):
    """lambda and delta from split-sample band powers.

    t1, t2: total band power, b1, b2: fitted aperiodic band power, each of
    shape (n_subjects, 2) with column s the estimate from half s (three
    columns, one per third, with instrument="third"), for conditions 1 and 2. The profiled moment is scanned over grid (default
    -0.5 to 1.5; outside it the weights become extreme and spurious roots
    appear). Returns dict(lam, delta, roots): lam is the root nearest `near`
    if given (used to track one root across bootstrap samples), otherwise
    the downward crossing, the direction of the moment at the true value,
    nearest the grid point where the moment is smallest in absolute value;
    nan if the moment does not change sign. instrument selects the half that
    supplies the instrument and the weights (see _stack), or "third" for
    three replicates (see _stack3).
    """
    grid = np.linspace(-0.5, 1.5, 81) if grid is None else grid
    arrs = _stack(t1, t2, b1, b2, instrument)
    f = lambda g: _m2(g, *arrs)[0]
    vals = np.array([f(g) for g in grid])
    roots, down = [], []
    for i in range(grid.size - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(brentq(f, grid[i], grid[i + 1]))
            down.append(vals[i] > 0)
    if not roots:
        return dict(lam=np.nan, delta=np.nan, roots=[])
    if near is not None:
        lam = min(roots, key=lambda r: abs(r - near))
    else:
        cand = [r for r, d in zip(roots, down) if d] or roots
        j = int(np.nanargmin(np.abs(vals)))
        lam = min(cand, key=lambda r: abs(r - grid[j]))
    ed = _m2(lam, *arrs)[1]
    return dict(lam=float(lam), delta=float(np.log(ed)) if ed > 0 else np.nan,
                roots=[float(r) for r in roots])


def _levels_moments(lam, gamma, T, B, Z, X, W, groups, starts):
    """Moments of the within-unit levels model (see estimate_levels)."""
    Wl = W ** (lam - 1.0)
    r = (T - B) / B ** lam
    g = np.exp(X @ gamma) if X.shape[1] else np.ones_like(r)
    num = np.add.reduceat(Wl * r, starts)
    den = np.add.reduceat(Wl * g, starts)
    c = (num / np.where(den == 0, np.nan, den))[groups]
    e = Wl * (r - c * g)
    m = [np.sum(e * Z)] + [np.sum(e * X[:, k]) for k in range(X.shape[1])]
    return np.array(m) / np.sum(Wl)


def estimate_levels(T, B, Bz, groups, X=None, grid=None):
    """Coupling from fluctuations of band power within units (e.g. epochs
    within a participant), without logging the periodic power.

    Model: a_ij = c_i exp(gamma' x_ij) b_ij^lambda u_ij with E[u - 1] = 0,
    where i indexes units (participants), j epochs, and x are measured
    confounders (arousal, eye, muscle proxies; centred within unit). Moments:
    the residual a/b^lambda - c_i exp(gamma' x) is uncorrelated with the
    instrument z (ln Bz centred within unit and residualised on x) and with
    each x; c_i is profiled per unit.

    T: total band power from one estimate (e.g. taper A); B: aperiodic band
    power from an independent estimate (taper B), used inside a = T - B and
    b^lambda; Bz: aperiodic band power from a third estimate (taper C), used
    for the instrument and the weights. groups: unit label per epoch.
    """
    grid = np.linspace(-0.5, 1.5, 81) if grid is None else grid
    order = np.argsort(groups, kind="stable")
    T, B, Bz, groups = T[order], B[order], Bz[order], np.asarray(groups)[order]
    X = np.zeros((T.size, 0)) if X is None else np.asarray(X, float)[order]
    _, starts, inv = np.unique(groups, return_index=True, return_inverse=True)
    # centre x within units; instrument = ln Bz centred within units, residualised on x
    def centre(v):
        m = np.add.reduceat(v, starts) / np.diff(np.r_[starts, v.size])
        return v - m[inv]
    Xc = np.column_stack([centre(X[:, k]) for k in range(X.shape[1])]) if X.shape[1] else X
    z = centre(np.log(Bz))
    if X.shape[1]:
        z = z - Xc @ np.linalg.lstsq(Xc, z, rcond=None)[0]
    k = Xc.shape[1]

    def profile(lam):
        if k == 0:
            return _levels_moments(lam, np.zeros(0), T, B, z, Xc, Bz, inv, starts)[0], np.zeros(0)
        sol = optimize.least_squares(
            lambda gm: _levels_moments(lam, gm, T, B, z, Xc, Bz, inv, starts)[1:],
            np.zeros(k))
        return _levels_moments(lam, sol.x, T, B, z, Xc, Bz, inv, starts)[0], sol.x

    vals = np.array([profile(g)[0] for g in grid])
    roots = [brentq(lambda l: profile(l)[0], grid[i], grid[i + 1])
             for i in range(grid.size - 1)
             if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0]
    if not roots:
        return dict(lam=np.nan, gamma=np.full(k, np.nan), roots=[])
    j = int(np.nanargmin(np.abs(vals)))
    lam = min(roots, key=lambda r: abs(r - grid[j]))
    return dict(lam=float(lam), gamma=profile(lam)[1], roots=[float(r) for r in roots])


def bootstrap(t1, t2, b1, b2, nboot=300, rng=None, grid=None, instrument="total"):
    """Point estimate plus subject-bootstrap percentile interval.

    Each bootstrap sample takes the root nearest the full-sample estimate;
    boot_fail is the share of samples in which the moment had no root.
    grid and instrument are passed to estimate().
    """
    rng = rng or np.random.default_rng(0)
    est = estimate(t1, t2, b1, b2, grid=grid, instrument=instrument)
    n = t1.shape[0]
    near = est["lam"] if np.isfinite(est["lam"]) else None
    bs = np.array([estimate(t1[i], t2[i], b1[i], b2[i], grid=grid, near=near,
                            instrument=instrument)["lam"]
                   for i in (rng.integers(0, n, n) for _ in range(nboot))])
    ok = np.isfinite(bs)
    est.update(ci=tuple(np.percentile(bs[ok], [2.5, 97.5])) if ok.sum() > 10
               else (np.nan, np.nan),
               boot_sd=float(np.std(bs[ok])) if ok.sum() > 10 else np.nan,
               boot_fail=float(1 - ok.mean()))
    return est
