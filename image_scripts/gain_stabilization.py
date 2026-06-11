"""
gain_stabilization.py — per-spectrum gain-drift detection and correction.

Context: the rooftop detector's HV/gain stabilization can fail. Two distinct
failure modes have been observed (see docs/todo.md "Detector gain drift" and the
logbook):

  - GAIN DRIFT (e.g. 2025-04-23): the photopeaks slide smoothly in channel
    space over hours while total counts are conserved. RECOVERABLE — find the
    per-spectrum calibration and resample onto the nominal grid.
  - COLLAPSE (e.g. 2026-05-13, 2026-06-10): the response above a few hundred
    keV drops out entirely; counts are lost, not relocated. NOT recoverable in
    software — this module returns None for such spectra (the honest answer).

This module finds a per-spectrum linear energy calibration E = a + b*ch from
the natural background lines, without assuming any prior calibration is valid
(validated against real data where the digitization changed 4096 -> 8192
channels between years, i.e. the "calibration" can be off by 2x).

Method (each choice below was forced by a real failure during validation):

  1. Find candidate peaks and rank them by PROMINENCE, not height. Wiggles
     riding the tall low-energy continuum out-rank real photopeaks on height
     and crowd them out of the candidate list; photopeaks win on prominence.
  2. Try assigning 3 candidates to the strong reference lines (Pb-214 352,
     Bi-214 609, Bi-214 1120, K-40 1461) and keep the lowest-residual linear
     fit whose slope lies within ±drift_tol of the channel-count-expected
     value (the digitization fixes full scale, so b ~ nominal_b * ref_ch/n_ch)
     and whose offset is small. The slope band is what kills wrong assignments:
     with it open (0.2-1.3 keV/ch) the finder confidently mis-assigned most
     spectra and compressed them to low energy.
  3. Tl-208 2614 NEVER enters the fit. Its centroid is too noisy per-300s and
     it has the longest lever arm, so a bad centroid drags the slope hardest.
     It is confirm-only: a fit gains rank if any detected peak lands within
     tl_confirm_keV of where the fit predicts 2614.
  4. If no 3-peak fit exists, a 2-peak fallback runs with a much tighter slope
     band (±two_peak_drift_tol). Safe because the strong 352/609 pair maps to
     wrong line pairs only at slopes far outside that band.
  5. If nothing passes, return None — an honest gap, never a guess.

Validation (2026-06-11, raw 300 s CNFs):
  - 2025-04-23 (4096 ch, real drift): 280/280 fit, slope median 0.696
    IQR [0.677, 0.724]; drifting peaks straighten to vertical lines at their
    true energies in the corrected waterfall.
  - 2026-06-08 (8192 ch, stable):    271/278 fit, slope median 0.3325
    IQR [0.3319, 0.3330] vs nominal 0.3322 — the correction is a no-op on
    stable data. The 7 refusals are midday spectra with visibly degraded
    resolution (early signs of the collapse failure mode).

Dependencies: numpy; scipy (scipy.signal.find_peaks, imported lazily inside
fit_spectrum_gain). scipy is in environment.yml — on a server env created
before 2026-06-11, run `conda install scipy` once.

Public API:
  fit_collection_gains(spectra, nominal_cal, cfg=None) -> list[dict | None]
      (the production entry point: per-spectrum fits + temporal-consistency pass)
  fit_spectrum_gain(counts, nominal_cal, cfg=None) -> dict | None
  correct_counts(counts, gain_calibration, nominal_cal) -> np.ndarray
  DEFAULT_CONFIG
"""

import numpy as np

# Strong natural-background lines used IN the calibration fit (keV).
REF_FIT = [351.9, 609.3, 1120.3, 1460.8]
# Confirm-only line (keV): votes for a candidate fit, never enters the fit.
TL208 = 2614.5

DEFAULT_CONFIG = {
    # Slope band for the 3-peak fit, as a fraction of the channel-count-expected
    # slope. Wide enough for real drift (2025-04-23 reached ~+9%), tight enough
    # to kill wrong line assignments.
    'drift_tol': 0.25,
    # Much tighter band for the 2-peak fallback (a 2-point fit has no residual
    # to gate on, so the band does all the work).
    'two_peak_drift_tol': 0.12,
    # Gates on the 3-peak fit.
    'max_rms_keV': 8.0,
    'max_offset_keV': 60.0,
    # A detected peak within this of the predicted 2614 channel confirms a fit.
    'tl_confirm_keV': 40.0,
    # Peak finding: boxcar smoothing width, minimum prominence as a fraction of
    # the smoothed spectrum maximum, minimum peak separation, and the channel
    # where the search starts (skips the electronic-noise rise at the bottom).
    'smooth_ch': 21,
    'min_prominence_frac': 0.01,
    'min_peak_distance_ch': 50,
    'search_start_ch': 60,
    # Candidate pool sizes (top-N by prominence).
    'n_candidates': 12,
    'n_candidates_2peak': 5,
    # Channel count the nominal calibration's slope refers to. The digitization
    # fixes full scale, so at n_ch channels the expected slope is
    # nominal_b * ref_channels / n_ch (e.g. 0.332 at 8192 -> 0.664 at 4096).
    'ref_channels': 8192,
    # Center of the slope band. None -> computed from nominal_cal/ref_channels.
    # The temporal second pass sets this to the local rolling-median slope.
    'expected_b': None,
    # --- fit_collection_gains (temporal-consistency pass) ---
    # Rolling window (spectra, ~1h at 12/h) for the local median slope, the
    # fractional deviation from it that marks a first-pass fit as an outlier,
    # and the slope band half-width used when re-fitting outliers/refusals.
    'temporal_window': 12,
    'temporal_max_dev': 0.02,
    'temporal_refit_tol': 0.02,
}


def fit_spectrum_gain(counts, nominal_cal, cfg=None):
    """Fit a per-spectrum linear energy calibration from background lines.

    Calibration-agnostic: nominal_cal is used only to set the physically
    expected slope (via ref_channels scaling) and as the reference the
    gain_ratio is reported against — the peak search itself assumes nothing.

    Parameters
    ----------
    counts : 1-D array of channel counts (any channel count; 4096 and 8192 tested).
    nominal_cal : [c0, c1, c2, c3] — the standard calibration downstream ROIs
        assume, with c1 the slope at cfg['ref_channels'] channels.
    cfg : optional dict overriding DEFAULT_CONFIG keys.

    Returns
    -------
    dict with keys:
        calibration  : fitted [a, b, 0.0, 0.0] for THIS spectrum's channel axis
        gain_ratio   : fitted slope / expected slope at this channel count
                       (1.0 == no drift; this is the value to trend over time)
        rms_keV      : residual RMS of the fit anchors (0.0 for 2-peak fits)
        n_fit_peaks  : 3 or 2 (2 == fallback tier)
        tl_confirmed : True if a detected peak sits where the fit predicts 2614
    or None — no trustworthy fit (drop/skip the spectrum; on collapse-mode
    spectra this is the correct outcome, there is nothing to recover).
    """
    from itertools import combinations
    from scipy.signal import find_peaks

    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    counts = np.asarray(counts, dtype=float)
    n_ch = len(counts)
    expected_b = cfg['expected_b']
    if expected_b is None:
        expected_b = nominal_cal[1] * cfg['ref_channels'] / n_ch
    b_lo = expected_b * (1 - cfg['drift_tol'])
    b_hi = expected_b * (1 + cfg['drift_tol'])
    start = cfg['search_start_ch']

    sm = np.convolve(counts, np.ones(cfg['smooth_ch']) / cfg['smooth_ch'], 'same')
    if sm[start:].max() <= 0:
        return None
    pk, props = find_peaks(sm[start:],
                           prominence=sm[start:].max() * cfg['min_prominence_frac'],
                           distance=cfg['min_peak_distance_ch'])
    pk = pk + start
    if len(pk) < 2:
        return None
    prom_order = np.argsort(props['prominences'])[::-1]
    cand = np.sort(pk[prom_order[:cfg['n_candidates']]]).tolist()
    pk_all = np.asarray(pk, dtype=float)   # ALL peaks: Tl-208 is weak and may
                                           # not make the top-N candidate pool

    def tl_confirmed(a, b):
        ch_tl = (TL208 - a) / b
        return (0 < ch_tl < n_ch) and \
            bool(np.any(np.abs((a + b * pk_all) - TL208) < cfg['tl_confirm_keV']))

    best = None   # ranked by (n_fit_peaks, tl_confirmed, -rms)

    if len(cand) >= 3:
        for cc in combinations(cand, 3):
            for rc in combinations(REF_FIT, 3):
                b, a = np.polyfit(cc, rc, 1)
                if not (b_lo <= b <= b_hi) or abs(a) > cfg['max_offset_keV']:
                    continue
                rms = float(np.sqrt(np.mean((a + b * np.array(cc) - np.array(rc)) ** 2)))
                if rms > cfg['max_rms_keV']:
                    continue
                score = (3, tl_confirmed(a, b), -rms)
                if best is None or score > best[0]:
                    best = (score, (a, b), rms)

    if best is None:
        b2_lo = expected_b * (1 - cfg['two_peak_drift_tol'])
        b2_hi = expected_b * (1 + cfg['two_peak_drift_tol'])
        strong = np.sort(pk[prom_order[:cfg['n_candidates_2peak']]]).tolist()
        for cc in combinations(strong, 2):
            for rc in combinations(REF_FIT, 2):
                b, a = np.polyfit(cc, rc, 1)
                if not (b2_lo <= b <= b2_hi) or abs(a) > cfg['max_offset_keV']:
                    continue
                score = (2, tl_confirmed(a, b), 0.0)
                if best is None or score > best[0]:
                    best = (score, (a, b), 0.0)

    if best is None:
        return None
    score, (a, b), rms = best
    # gain_ratio is always vs the NOMINAL expectation, even when the search band
    # was centered elsewhere (temporal second pass) — it's the trended quantity.
    nominal_expected_b = nominal_cal[1] * cfg['ref_channels'] / n_ch
    return {
        'calibration': [float(a), float(b), 0.0, 0.0],
        'gain_ratio': float(b / nominal_expected_b),
        'rms_keV': float(rms),
        'n_fit_peaks': int(score[0]),
        'tl_confirmed': bool(score[1]),
    }


def fit_collection_gains(spectra, nominal_cal, cfg=None):
    """Fit gains for a time-ordered sequence of spectra, with a temporal-
    consistency second pass. This is the production entry point.

    Physical prior: gain is continuous — it cannot jump several percent between
    adjacent 5-minute spectra. So after the independent per-spectrum fits:

      1. Build the rolling-median slope over fitted neighbors.
      2. First-pass fits whose slope deviates from the local median by more than
         temporal_max_dev are discarded (they passed the static gates but are
         physically impossible — the off-track outlier class).
      3. Outliers AND first-pass refusals get one re-fit with the slope band
         centered on the local median, half-width temporal_refit_tol, and the
         2-peak tier allowed to use the full candidate pool (safe: with a ±2%
         band even a 2-line assignment is unambiguous). This rescues spectra
         where one of the strong lines fell out of the candidate pool.
      4. Whatever still fails stays None — on collapse-mode spectra that is the
         correct, honest outcome.

    Parameters
    ----------
    spectra : sequence of 1-D count arrays, in time order, same channel count.
    nominal_cal, cfg : as for fit_spectrum_gain.

    Returns
    -------
    list of (dict | None), same length/order as spectra. Re-fitted entries are
    marked with 'refit': True.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    n = len(spectra)
    results = [fit_spectrum_gain(s, nominal_cal, cfg) for s in spectra]

    bs = np.array([r['calibration'][1] if r else np.nan for r in results])
    if np.isnan(bs).all():
        return results

    half = max(1, cfg['temporal_window'] // 2)

    def local_median(i):
        win = bs[max(0, i - half):i + half + 1]
        win = win[~np.isnan(win)]
        return float(np.median(win)) if len(win) else np.nan

    med = np.array([local_median(i) for i in range(n)])
    outlier = ~np.isnan(bs) & ~np.isnan(med) & \
        (np.abs(bs - med) > cfg['temporal_max_dev'] * med)

    # Discard outliers first so they don't contaminate the re-fit medians.
    for i in np.where(outlier)[0]:
        results[i] = None
        bs[i] = np.nan
    med = np.array([local_median(i) for i in range(n)])

    refit_cfg_base = {**cfg,
                      'drift_tol': cfg['temporal_refit_tol'],
                      'two_peak_drift_tol': cfg['temporal_refit_tol'],
                      'n_candidates_2peak': cfg['n_candidates']}
    for i in range(n):
        if results[i] is not None or np.isnan(med[i]):
            continue
        r = fit_spectrum_gain(spectra[i], nominal_cal,
                              {**refit_cfg_base, 'expected_b': med[i]})
        if r is not None:
            r['refit'] = True
            results[i] = r
    return results


def correct_counts(counts, gain_calibration, nominal_cal):
    """Resample a drifted spectrum onto the nominal energy grid, conserving counts.

    counts[i] truly belongs at energy gain_calibration(i); this redistributes the
    counts so that output channel j holds what belongs at nominal energy
    nominal_cal(j). Peaks land back on their nominal channels; total counts are
    preserved (cumulative-interpolation rebin). Input and output have the same
    channel count — run AFTER standardize_channel_counts when used in the
    pipeline, so nominal_cal's channel axis matches.
    """
    def energy_at(channels, cal):
        ch = np.asarray(channels, dtype=float)
        c0, c1, c2, c3 = cal
        return c0 + c1 * ch + c2 * ch ** 2 + c3 * ch ** 3

    counts = np.asarray(counts, dtype=float)
    n = len(counts)
    edges = np.arange(n + 1) - 0.5

    src_edge_E = energy_at(edges, gain_calibration)
    tgt_edge_E = energy_at(edges, nominal_cal)

    cum = np.concatenate([[0.0], np.cumsum(counts)])
    cum_at_tgt = np.interp(tgt_edge_E, src_edge_E, cum, left=0.0, right=cum[-1])
    corrected = np.diff(cum_at_tgt)
    corrected[corrected < 0] = 0.0
    return corrected
