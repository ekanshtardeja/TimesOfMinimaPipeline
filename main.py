#!/usr/bin/env python3
"""
tess_eb_minima.py
-----------------
Primary-minimum timings (ToM) of an eclipsing binary from every TESS light curve
available on MAST.

Pipeline
  1. Ask for / accept a star name and search MAST (lightkurve) for all TESS light curves.
  2. Group the files by cadence and give each a rating out of 10
     (120 s = good -> 10, 600 s = poor -> 4; table below, edit freely).
  3. Find the orbital period with BLS on the best-rated file, then resolve the
     classic EB ambiguity (BLS returns P/2 when primary and secondary are similar)
     using a secondary-eclipse test and an odd/even depth test.
  4. In every file, locate the primary eclipses and fit
        f(t) = c0 + c1*(t - tc) - A * exp(-(t - mu)^2 / (2 s^2))
     to each one. mu = time of minimum. Each ToM inherits its file's cadence rating.
  5. Write a CSV of all minima, one diagnostic PNG per file, and an O-C plot.

Usage
  python tess_eb_minima.py                       # prompts for star name
  python tess_eb_minima.py "CW Eri"
  python tess_eb_minima.py "TIC 98853987" --period 2.7284 --t0 2458413.4527
  python tess_eb_minima.py "CW Eri" --authors SPOC QLP --min-per-file 5

Requirements: pip install lightkurve astropy scipy numpy pandas matplotlib
Times are BJD_TDB (TESS BTJD + 2457000).
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")

BTJD0 = 2457000.0

# (upper limit on exposure time in seconds, rating /10)
CADENCE_RATINGS = [
    (30, 10),       # 20 s  fast cadence
    (150, 10),      # 120 s 2-min cadence           -> good
    (250, 7),       # 200 s FFI (Cycle 5+)
    (700, 4),       # 600 s 10-min FFI (Cycles 3-4) -> poor
    (np.inf, 2),    # 1800 s 30-min FFI (Cycles 1-2)
]


def cadence_rating(exptime_s):
    for lim, r in CADENCE_RATINGS:
        if exptime_s <= lim:
            return r
    return 1


def cadence_label(exptime_s):
    return f"{int(round(exptime_s))}s"


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def fetch_lightcurves(target, authors=None, flux_column=None, quality_bitmask="default"):
    import lightkurve as lk

    print(f"\nSearching MAST for TESS light curves of '{target}' ...")
    sr = lk.search_lightcurve(target, mission="TESS")
    if len(sr) == 0:
        sys.exit("No TESS light curves found. Try a TIC ID, e.g. 'TIC 123456789'.")

    wanted = [a.upper() for a in authors] if authors else None
    entries = []
    for i in range(len(sr)):
        author = str(sr.author[i])
        if wanted and author.upper() not in wanted:
            continue
        exptime = float(np.atleast_1d(sr.exptime[i].value)[0])
        mission = str(sr.mission[i])
        m = re.search(r"(\d+)", mission)
        sector = int(m.group(1)) if m else -1
        kw = {"quality_bitmask": quality_bitmask}
        if flux_column:
            kw["flux_column"] = flux_column
        try:
            lc = sr[i].download(**kw)
            if lc is None:
                raise ValueError("empty download")
        except Exception as exc:
            print(f"  [skip] Sector {sector:>3} {author:<10} {cadence_label(exptime):>6}: {exc}")
            continue
        entries.append(dict(sector=sector, author=author, exptime=exptime,
                            rating=cadence_rating(exptime), lc=lc))
        print(f"  [ok]   Sector {sector:>3} {author:<10} {cadence_label(exptime):>6}"
              f"  rating {cadence_rating(exptime)}/10")
    if not entries:
        sys.exit("Nothing could be downloaded.")
    return entries


def clean_arrays(lc):
    """Return t (BTJD, days), normalised flux, flux error, median cadence (days)."""
    lc = lc.remove_nans().normalize()
    lc = lc.remove_outliers(sigma_lower=np.inf, sigma_upper=4)   # keep eclipses
    t = np.asarray(lc.time.tdb.jd, float) - BTJD0
    f = np.asarray(np.ma.filled(lc.flux.value, np.nan), float)
    try:
        e = np.asarray(np.ma.filled(lc.flux_err.value, np.nan), float)
    except Exception:
        e = np.full_like(f, np.nan)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f, e = t[ok], f[ok], e[ok]
    o = np.argsort(t)
    t, f, e = t[o], f[o], e[o]
    scatter = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2) if len(f) > 3 else 1e-3
    bad = ~np.isfinite(e) | (e <= 0)
    if bad.all():
        e[:] = scatter
    else:
        e[bad] = np.nanmedian(e[~bad])
    cad = np.median(np.diff(t)) if len(t) > 1 else 0.0
    return t, f, e, cad


# ----------------------------------------------------------------------------
# Period / ephemeris
# ----------------------------------------------------------------------------
def bin_series(t, f, dt):
    k = np.floor((t - t.min()) / dt).astype(int)
    df = pd.DataFrame({"k": k, "t": t, "f": f}).groupby("k").mean()
    return df["t"].values, df["f"].values


def run_bls(t, f, pmin, pmax):
    if np.median(np.diff(t)) < 0.006:          # finer than ~8.6 min -> bin to 10 min
        t, f = bin_series(t, f, 10 / 1440)
    baseline = t.max() - t.min()
    pmax = min(pmax, baseline / 2)
    if pmax <= pmin:
        raise RuntimeError("Light curve too short for a period search; supply --period.")
    bls = BoxLeastSquares(t, f)
    best = None
    passes = [(pmin, min(1.0, pmax), [0.01, 0.02, 0.03, 0.045, 0.065, 0.09, 0.13, 0.18]),
              (1.0, pmax, [0.04, 0.07, 0.1, 0.15, 0.22, 0.32, 0.45, 0.6])]
    for lo, hi, durs in passes:
        if hi <= lo:
            continue
        durs = np.array([d for d in durs if d < 0.9 * lo])
        if durs.size == 0:
            continue
        periods = bls.autoperiod(durs, minimum_period=lo, maximum_period=hi, frequency_factor=1.0)
        res = bls.power(periods, durs)
        j = int(np.argmax(res.power))
        cand = dict(power=res.power[j], P=res.period[j], t0=res.transit_time[j],
                    dur=res.duration[j], depth=res.depth[j])
        if best is None or cand["power"] > best["power"]:
            best = cand
    # local refinement around the peak
    P0 = best["P"]
    periods = np.linspace(P0 * 0.998, P0 * 1.002, 4001)
    res = bls.power(periods, best["dur"])
    j = int(np.argmax(res.power))
    best.update(P=res.period[j], t0=res.transit_time[j], depth=res.depth[j])
    return best


def phase(t, P, t0):
    return ((t - t0) / P + 0.5) % 1.0 - 0.5          # primary at 0, secondary at +-0.5


def depth_at(t, f, P, t0, centre, hw):
    ph = phase(t, P, t0)
    d = np.abs(((ph - centre) + 0.5) % 1.0 - 0.5)
    inside = d < hw
    d0 = np.abs(ph)
    d5 = np.abs(np.abs(ph) - 0.5)
    out = (d0 > 2 * hw) & (d5 > 2 * hw)
    if inside.sum() < 3 or out.sum() < 10:
        return 0.0, np.inf
    base = np.median(f[out])
    sig = 1.4826 * np.median(np.abs(f[out] - base))
    return base - np.median(f[inside]), sig / np.sqrt(inside.sum())


def resolve_period(t, f, P, t0, dur):
    """Fix the P vs 2P ambiguity of BLS for eclipsing binaries."""
    hw = max(0.25 * dur / P, 0.002)
    dp, sp = depth_at(t, f, P, t0, 0.0, hw)
    ds, ss = depth_at(t, f, P, t0, 0.5, hw)
    if ds > 5 * ss:
        note = f"secondary eclipse seen at phase 0.5 (depth {ds:.4f} vs {dp:.4f}) -> P accepted"
        if ds > dp:
            t0 += P / 2
        return P, t0, note
    P2 = 2 * P
    hw2 = hw / 2
    de, se = depth_at(t, f, P2, t0, 0.0, hw2)
    do, so = depth_at(t, f, P2, t0, 0.5, hw2)
    diff = abs(de - do)
    if diff > 5 * np.hypot(se, so) and diff > 0.03 * max(de, do):
        if do > de:
            t0 += P
        return P2, t0, (f"odd/even depths differ ({de:.4f} vs {do:.4f}) -> BLS gave P/2, "
                        f"period doubled")
    return P, t0, ("no secondary at phase 0.5 and odd/even depths equal: if the two eclipses "
                   "are near-identical the true period may be 2P; pass --period to override")


def eclipse_fwhm(t, f, P, t0, cad, dur_guess):
    """FWHM (days) of the primary eclipse from the folded, binned light curve."""
    ph = phase(t, P, t0)
    nb = int(np.clip(P / max(cad, dur_guess / 15), 100, 3000))
    edges = np.linspace(-0.5, 0.5, nb + 1)
    idx = np.digitize(ph, edges) - 1
    prof = np.array([np.median(f[idx == k]) if np.any(idx == k) else np.nan for k in range(nb)])
    cen = 0.5 * (edges[1:] + edges[:-1])
    good = np.isfinite(prof)
    prof = np.interp(cen, cen[good], prof[good])
    base = np.median(prof)
    k0 = int(np.argmin(np.abs(cen)))
    win = max(2, int(0.1 * nb))
    k0 = k0 - win + int(np.argmin(prof[max(0, k0 - win):k0 + win]))
    half = prof[k0] + 0.5 * (base - prof[k0])
    l = k0
    while l > 0 and prof[l] < half:
        l -= 1
    r = k0
    while r < nb - 1 and prof[r] < half:
        r += 1
    fwhm = (cen[r] - cen[l]) * P
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = 0.6 * dur_guess
    return fwhm


# ----------------------------------------------------------------------------
# Minima extraction
# ----------------------------------------------------------------------------
def gauss_model(t, c0, c1, A, mu, s, tc):
    return c0 + c1 * (t - tc) - A * np.exp(-0.5 * ((t - mu) / s) ** 2)


def local_anchor(t, f, P, t0, fwhm):
    """Time of a primary eclipse inside this file (deepest point of the local fold)."""
    ph = phase(t, P, t0)
    bw = max(fwhm / P / 4, 1e-4)
    nb = max(int(1 / bw), 50)
    idx = np.clip(((ph + 0.5) * nb).astype(int), 0, nb - 1)
    prof = np.full(nb, np.nan)
    for k in np.unique(idx):
        prof[k] = np.median(f[idx == k])
    sm = median_filter(np.nan_to_num(prof, nan=np.nanmedian(prof)), size=3, mode="wrap")
    kmin = int(np.argmin(sm))
    ph_min = (kmin + 0.5) / nb - 0.5
    return t0 + ph_min * P


def fit_minimum(t, f, e, tc, w, fwhm, cad):
    sel = np.abs(t - tc) < w
    n = int(sel.sum())
    if n < 8:
        return None, "too few points"
    tt, ff, ee = t[sel], f[sel], e[sel]
    k = max(1, min(5, n // 8)) | 1
    sm = median_filter(ff, size=k)
    j = int(np.argmin(sm))
    edge = np.abs(tt - tc) > 0.6 * w
    c0 = np.median(ff[edge]) if edge.sum() >= 3 else np.max(sm)
    A0 = max(c0 - sm[j], 1e-4)
    s0 = max(fwhm / 2.355, cad)
    p0 = [c0, 0.0, A0, tt[j], s0]
    lo = [0.0, -np.inf, 0.0, tc - w, cad / 3]
    hi = [2.0, np.inf, 2.0, tc + w, w]
    try:
        popt, pcov = curve_fit(lambda x, c0, c1, A, mu, s: gauss_model(x, c0, c1, A, mu, s, tc),
                               tt, ff, p0=p0, sigma=ee, absolute_sigma=True,
                               bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None, "fit failed"
    resid = (ff - gauss_model(tt, *popt, tc)) / ee
    chi2r = float(np.sum(resid ** 2) / max(n - 5, 1))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf)) * np.sqrt(max(chi2r, 1.0))
    c0, c1, A, mu, s = popt
    A_err, mu_err = perr[2], perr[3]

    if not np.isfinite(mu_err) or mu_err <= 0:
        return None, "bad covariance"
    if abs(mu - tc) > 0.7 * w:
        return None, "minimum at window edge"
    if A_err == 0 or A / A_err < 5:
        return None, "low depth S/N"
    if mu_err > s / 2:
        return None, "timing error too large"
    near = np.abs(tt - mu) < max(fwhm / 2, 1.5 * cad)
    if (tt[near] < mu).sum() < 2 or (tt[near] > mu).sum() < 2:
        return None, "eclipse not covered on both sides"
    core = tt[np.abs(tt - mu) < max(fwhm, 2 * cad)]
    if core.size < 4 or np.max(np.diff(core)) > max(2.5 * cad, fwhm / 2):
        return None, "data gap inside eclipse"

    return dict(mu=mu, mu_err=mu_err, A=A, A_err=A_err, s=s, c0=c0, c1=c1, tc=tc,
                chi2r=chi2r, n=n, popt=popt), "ok"


def process_file(entry, P, t0, fwhm, min_per_file):
    t, f, e, cad = entry["t"], entry["f"], entry["e"], entry["cad"]
    w = float(np.clip(1.5 * fwhm, 6 * cad, 0.2 * P))
    tref = local_anchor(t, f, P, t0, fwhm)
    n0 = int(np.ceil((t.min() - w - tref) / P))
    n1 = int(np.floor((t.max() + w - tref) / P))
    minima, reasons = [], {}
    for n in range(n0, n1 + 1):
        tc = tref + n * P
        if tc < t.min() - 0.5 * w or tc > t.max() + 0.5 * w:
            continue
        res, why = fit_minimum(t, f, e, tc, w, fwhm, cad)
        if res is None:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            minima.append(res)
    entry["minima"], entry["rejects"], entry["window"] = minima, reasons, w
    tag = f"S{entry['sector']:>3} {entry['author']:<10} {cadence_label(entry['exptime']):>6}"
    msg = f"  {tag}  rating {entry['rating']:>2}/10  ->  {len(minima)} primary minima"
    if reasons:
        msg += "  (rejected: " + ", ".join(f"{v} {k}" for k, v in reasons.items()) + ")"
    if len(minima) < min_per_file:
        msg += f"  [WARNING: fewer than {min_per_file}]"
    print(msg)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def fit_ephemeris(T, err, P):
    T0 = T[np.argmin(err)]
    for _ in range(3):
        E = np.round((T - T0) / P)
        if len(np.unique(E)) < 2:
            return T0, P, E
        wgt = 1 / err ** 2
        A = np.vstack([np.ones_like(E), E]).T
        cov = np.linalg.inv(A.T @ (A * wgt[:, None]))
        T0, P = cov @ (A.T @ (wgt * T))
    return T0, P, np.round((T - T0) / P)


def plot_file(entry, P, outdir, star):
    t, f, mins = entry["t"], entry["f"], entry["minima"]
    nm = len(mins)
    ncol = min(5, max(nm, 1))
    nrow = 1 + int(np.ceil(nm / ncol)) if nm else 1
    fig = plt.figure(figsize=(3.2 * ncol + 1, 2.8 * nrow + 0.5))
    ax = fig.add_subplot(nrow, 1, 1)
    ax.plot(t, f, ".", ms=1.5, color="0.3")
    for m in mins:
        ax.axvline(m["mu"], color="crimson", lw=0.8, alpha=0.7)
    ax.set_xlabel("BTJD (BJD_TDB - 2457000)")
    ax.set_ylabel("Norm. flux")
    ax.set_title(f"{star} | Sector {entry['sector']} | {entry['author']} | "
                 f"{cadence_label(entry['exptime'])} | rating {entry['rating']}/10 | "
                 f"{nm} primary minima | P = {P:.6f} d")
    w = entry["window"]
    for i, m in enumerate(mins):
        a = fig.add_subplot(nrow, ncol, ncol + i + 1)
        sel = np.abs(t - m["tc"]) < w
        tt = t[sel]
        a.plot((tt - m["mu"]) * 24, f[sel], ".", ms=3, color="0.3")
        xx = np.linspace(tt.min(), tt.max(), 400)
        a.plot((xx - m["mu"]) * 24, gauss_model(xx, *m["popt"], m["tc"]), color="crimson", lw=1.2)
        a.axvline(0, color="royalblue", lw=0.8, ls="--")
        a.set_title(f"{m['mu'] + BTJD0:.5f}\n±{m['mu_err'] * 86400:.1f} s", fontsize=8)
        a.tick_params(labelsize=7)
        a.set_xlabel("hours from min", fontsize=7)
    fig.tight_layout()
    fn = os.path.join(outdir, f"S{entry['sector']:03d}_{entry['author']}_"
                              f"{cadence_label(entry['exptime'])}.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)


def plot_oc(df, outdir, star, P, T0):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sc = ax.scatter(df["T_min_BJD_TDB"] - BTJD0, df["O_minus_C_min"], c=df["cadence_rating"],
                    cmap="viridis", vmin=0, vmax=10, s=18, zorder=3)
    ax.errorbar(df["T_min_BJD_TDB"] - BTJD0, df["O_minus_C_min"], yerr=df["T_min_err_s"] / 60,
                fmt="none", ecolor="0.6", lw=0.8, zorder=2)
    ax.axhline(0, color="k", lw=0.6)
    plt.colorbar(sc, label="Cadence rating /10")
    ax.set_xlabel("BTJD")
    ax.set_ylabel("O - C (min)")
    ax.set_title(f"{star}: primary minima O-C  (T0 = {T0:.6f}, P = {P:.8f} d)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "O_minus_C.png"), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def analyse(star, entries, period=None, t0=None, pmin=0.25, pmax=30.0, min_per_file=4,
            outdir=None):
    outdir = outdir or re.sub(r"[^\w\-]+", "_", star.strip()) + "_TESS_minima"
    os.makedirs(outdir, exist_ok=True)

    for en in entries:
        en["t"], en["f"], en["e"], en["cad"] = clean_arrays(en["lc"])
    entries = [en for en in entries if len(en["t"]) > 100]
    entries.sort(key=lambda en: (en["sector"], en["exptime"], en["author"]))

    # summary by cadence
    print("\nFiles by cadence:")
    groups = {}
    for en in entries:
        groups.setdefault(en["exptime"], []).append(en)
    for ex in sorted(groups):
        secs = ", ".join(str(en["sector"]) for en in groups[ex])
        print(f"  {cadence_label(ex):>6}  rating {cadence_rating(ex):>2}/10  "
              f"{len(groups[ex])} file(s)  sectors: {secs}")

    # period on the best file (highest rating, then most points)
    best = max(entries, key=lambda en: (en["rating"], len(en["t"])))
    tb, fb, cb = best["t"], best["f"], best["cad"]
    if period:
        P = float(period)
        T0 = (float(t0) - BTJD0) if t0 else None
        if T0 is None:
            r = run_bls(tb, fb, 0.9 * P, 1.1 * P)
            T0, dur = r["t0"], r["dur"]
            note = "user period; epoch from BLS"
        else:
            dur, note = min(0.1 * P, 0.3), "user ephemeris"
        # make sure the epoch is on the deeper eclipse
        hw = max(0.25 * dur / P, 0.002)
        if depth_at(tb, fb, P, T0, 0.5, hw)[0] > depth_at(tb, fb, P, T0, 0.0, hw)[0]:
            T0 += P / 2
            note += " (epoch shifted to the deeper eclipse)"
    else:
        print(f"\nBLS period search on Sector {best['sector']} {best['author']} "
              f"{cadence_label(best['exptime'])} ...")
        r = run_bls(tb, fb, pmin, pmax)
        P, T0, note = resolve_period(tb, fb, r["P"], r["t0"], r["dur"])
        dur = r["dur"]
    fwhm = eclipse_fwhm(tb, fb, P, T0, cb, dur)
    print(f"  P = {P:.6f} d   T0 = {T0 + BTJD0:.5f} BJD_TDB   primary FWHM = {fwhm * 24:.2f} h")
    print(f"  {note}")

    print("\nFitting primary minima:")
    rows = []
    for en in entries:
        process_file(en, P, T0, fwhm, min_per_file)
        for m in en["minima"]:
            rows.append(dict(star=star, sector=en["sector"], author=en["author"],
                             cadence_s=int(round(en["exptime"])), cadence_rating=en["rating"],
                             T_min_BJD_TDB=m["mu"] + BTJD0, T_min_err_d=m["mu_err"],
                             T_min_err_s=m["mu_err"] * 86400, depth=m["A"],
                             depth_snr=m["A"] / m["A_err"], gauss_sigma_h=m["s"] * 24,
                             red_chi2=m["chi2r"], n_points=m["n"]))
        plot_file(en, P, outdir, star)

    if not rows:
        print("\nNo primary minima could be measured.")
        return None
    df = pd.DataFrame(rows).sort_values(["T_min_BJD_TDB", "cadence_rating"],
                                        ascending=[True, False]).reset_index(drop=True)
    T0f, Pf, E = fit_ephemeris(df["T_min_BJD_TDB"].values, df["T_min_err_d"].values, P)
    df.insert(5, "cycle", E.astype(int))
    df["O_minus_C_min"] = (df["T_min_BJD_TDB"] - (T0f + E * Pf)) * 1440
    csv = os.path.join(outdir, "primary_minima.csv")
    df.to_csv(csv, index=False, float_format="%.7f")
    plot_oc(df, outdir, star, Pf, T0f)

    print(f"\nLinear ephemeris from all minima: Min I = {T0f:.6f} + {Pf:.8f} E  (BJD_TDB)")
    print(f"Total primary minima: {len(df)} from {df.groupby(['sector', 'author', 'cadence_s']).ngroups}"
          f" file(s)")
    with pd.option_context("display.width", 160, "display.max_rows", 500,
                           "display.float_format", "{:.6f}".format):
        print(df[["sector", "author", "cadence_s", "cadence_rating", "cycle",
                  "T_min_BJD_TDB", "T_min_err_s", "O_minus_C_min"]].to_string(index=False))
    print(f"\nSaved: {csv}\n       per-file plots + O_minus_C.png in {outdir}/")
    return df


def main():
    ap = argparse.ArgumentParser(description="TESS primary-minimum timings for an eclipsing binary.")
    ap.add_argument("star", nargs="?", help="Star name or TIC ID (prompted if omitted)")
    ap.add_argument("--period", type=float, help="Known orbital period in days (skips BLS)")
    ap.add_argument("--t0", type=float, help="Known primary epoch, BJD_TDB")
    ap.add_argument("--pmin", type=float, default=0.25, help="Min BLS period, d (default 0.25)")
    ap.add_argument("--pmax", type=float, default=30.0, help="Max BLS period, d (default 30)")
    ap.add_argument("--authors", nargs="+", help="Keep only these pipelines, e.g. SPOC TESS-SPOC QLP")
    ap.add_argument("--flux-column", help="e.g. sap_flux (default: pipeline default, PDCSAP for SPOC)")
    ap.add_argument("--quality", default="default", help="lightkurve quality_bitmask")
    ap.add_argument("--min-per-file", type=int, default=4, help="Warn if a file gives fewer minima")
    ap.add_argument("--outdir", help="Output directory")
    a = ap.parse_args()

    star = a.star or input("Enter star name / TIC ID: ").strip()
    if not star:
        sys.exit("No star given.")
    entries = fetch_lightcurves(star, a.authors, a.flux_column, a.quality)
    analyse(star, entries, a.period, a.t0, a.pmin, a.pmax, a.min_per_file, a.outdir)


if __name__ == "__main__":
    main()
