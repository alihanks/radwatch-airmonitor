"""
gain_stabilization.py — per-spectrum gain-drift detection and correction.

Context: the rooftop detector's HV/gain stabilization drifts (daily, thermal),
sliding the photopeaks out of their fixed-channel ROI windows. The pipeline's
K-40 livetime filter misreads the resulting K-40 collapse as "bad livetime" and
drops the spectrum, losing ~6-10h of good data per day. See docs/todo.md
("Detector gain drift") and the logbook.

This module recovers that data: for each spectrum it locates natural background
anchor lines, fits a per-spectrum energy calibration, and resamples the spectrum
onto the standard energy grid so the peaks line back up. Downstream ROI logic
(and the K-40 filter) then see an aligned spectrum and keep the data.

Anchor strategy (built to survive gain drift, per design review):
  - The 609 keV Bi-214 line is the strongest environmental line but sits in a
    CROWDED neighborhood (Tl-208 583, annihilation 511, Cs-137 662), so under
    drift a naive search can grab the wrong peak. We therefore do NOT anchor
    identification on raw strength.
  - Bootstrap on the CLEAN, isolated singlets first: K-40 (1460.8) and
    Tl-208 (2614.5). Fit a coarse linear gain from those two (~1.2 MeV lever arm).
  - PREDICT where 609 should land under that coarse fit, look only in a narrow
    window there, and accept Bi-214 only if a peak sits at the predicted spot.
    Its strength then refines the low-energy end but can't hijack identification.
  - Geometric consistency gate: the final linear fit must have small residuals
    and a sane gain ratio, else we return None (caller drops the spectrum).

Public API:
  fit_spectrum_gain(counts, nominal_cal, cfg=None) -> dict | None
  correct_counts(counts, gain_calibration, nominal_cal) -> np.ndarray
  DEFAULT_CONFIG
"""

import numpy as np

try:
    from spectrum_calibration import energy_to_channel, calibrate_spectrum
except ImportError:  # when imported as image_scripts.gain_stabilization
    from image_scripts.spectrum_calibration import energy_to_channel, calibrate_spectrum


# Natural-background anchor lines (keV). 'isolation' is a reliability weight in
# [0,1]: how unambiguous the line is under drift (isolated singlet = 1.0,
# crowded = lower). Bi-214 609 is strong but crowded, so it's down-weighted.
ANCHORS = {
    'Bi-214': {'energy': 609.3,  'isolation': 0.5, 'clean': False},
    'K-40':   {'energy': 1460.8, 'isolation': 1.0, 'clean': True},
    'Tl-208': {'energy': 2614.5, 'isolation': 1.0, 'clean': True},
}

DEFAULT_CONFIG = {
    # Half-width of the search window for the CLEAN anchors, as a fraction of the
    # anchor's nominal channel (must cover the expected drift; the peak-finder
    # localizes on the strongest peak inside, so a wide window is safe for the
    # isolated K-40 / Tl-208 lines).
    'clean_window_frac': 0.09,
    # When localizing a peak inside a window, walk down each side to this fraction
    # of the peak height, then centroid that local region.
    'peak_floor_frac': 0.3,
    # Half-width (in channels) of the NARROW confirm window for Bi-214 around its
    # position predicted from the coarse fit.
    'confirm_window_ch': 30,
    # Minimum peak significance (net area / sqrt(gross)) to trust an anchor.
    'min_significance': 4.0,
    # Number of edge channels used to estimate the linear background under a peak.
    'bg_edge_frac': 0.12,
    # Geometric gate: reject the fit if residual RMS exceeds this (keV)...
    'max_rms_keV': 6.0,
    # ...or if the fitted gain (slope) drifts outside this band vs nominal.
    'min_gain_ratio': 0.85,
    'max_gain_ratio': 1.15,
}


def _energy_at(channels, cal):
    """Energy (keV) at fractional channel positions for cal=[c0,c1,c2,c3]."""
    ch = np.asarray(channels, dtype=float)
    c0, c1, c2, c3 = cal
    return c0 + c1 * ch + c2 * ch ** 2 + c3 * ch ** 3


def _peak_centroid(counts, lo, hi, cfg):
    """Locate the strongest peak in channel window [lo, hi] and centroid it.

    The window may be wide (to cover large drift) and may contain weaker
    neighbors, so we (1) subtract a linear background, (2) find the strongest
    net peak, (3) walk down to a fraction of its height on each side, and
    (4) centroid only that local region. This locks onto the dominant line
    instead of averaging the whole window.

    Returns dict(centroid, area, significance) or None.
    """
    n = len(counts)
    lo = max(0, int(lo))
    hi = min(n - 1, int(hi))
    if hi - lo < 6:
        return None
    seg = np.asarray(counts[lo:hi + 1], dtype=float)
    ch = np.arange(lo, hi + 1, dtype=float)

    # Linear background from the average of the window's edge channels.
    nbg = max(2, int(len(seg) * cfg['bg_edge_frac']))
    left = seg[:nbg].mean()
    right = seg[-nbg:].mean()
    bg = np.linspace(left, right, len(seg))
    net = seg - bg
    net[net < 0] = 0.0
    if net.sum() <= 0:
        return None

    # Strongest peak (light 3-channel smoothing for a stable location).
    sm = np.convolve(net, np.ones(3) / 3.0, mode='same')
    pk = int(np.argmax(sm))
    peak_h = net[pk]
    if peak_h <= 0:
        return None

    # Walk down each side to peak_floor_frac of the peak height.
    thr = cfg['peak_floor_frac'] * peak_h
    l = pk
    while l > 0 and net[l - 1] >= thr:
        l -= 1
    r = pk
    while r < len(net) - 1 and net[r + 1] >= thr:
        r += 1

    loc_ch = ch[l:r + 1]
    loc_net = net[l:r + 1]
    area = loc_net.sum()
    if area <= 0:
        return None
    centroid = float((loc_ch * loc_net).sum() / area)
    gross = seg[l:r + 1].sum()
    significance = area / np.sqrt(gross) if gross > 0 else 0.0
    return {'centroid': centroid, 'area': float(area), 'significance': float(significance)}


def fit_spectrum_gain(counts, nominal_cal, cfg=None):
    """Fit a per-spectrum linear energy calibration from background anchor lines.

    Parameters
    ----------
    counts : 1-D array of channel counts (standardized length, e.g. 8192).
    nominal_cal : list [c0,c1,c2,c3], the standard calibration the ROIs assume.
    cfg : optional config dict overriding DEFAULT_CONFIG.

    Returns
    -------
    dict with keys:
        calibration   : fitted [c0,c1,c2,c3] (linear; c2=c3=0)
        rms_keV       : residual RMS of the anchor fit
        n_anchors     : number of anchors used (2 or 3)
        anchors_used  : list of anchor names
        gain_ratio    : fitted slope / nominal slope (1.0 == no drift)
        k40_shift_keV : apparent shift of the K-40 line under the NOMINAL cal
                        (positive => peak moved up in apparent energy); None if
                        K-40 wasn't found
    or None if no trustworthy fit is possible (caller should drop the spectrum).
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    counts = np.asarray(counts, dtype=float)

    # --- Step 1: locate the clean, isolated anchors (K-40, Tl-208) ---
    found = {}  # name -> centroid channel
    k40_centroid = None
    for name in ('K-40', 'Tl-208'):
        E = ANCHORS[name]['energy']
        ch_nom = energy_to_channel(E, nominal_cal)
        half = cfg['clean_window_frac'] * ch_nom
        res = _peak_centroid(counts, ch_nom - half, ch_nom + half, cfg)
        if res and res['significance'] >= cfg['min_significance']:
            found[name] = res['centroid']
            if name == 'K-40':
                k40_centroid = res['centroid']

    # Need both clean anchors to bootstrap a coarse fit.
    if len(found) < 2:
        return None

    # --- Step 2: coarse linear fit from the clean anchors ---
    coarse_pts = [(found[n], ANCHORS[n]['energy']) for n in found]
    coarse_cal, _ = calibrate_spectrum(coarse_pts, order=1)

    # --- Step 3: predict & confirm Bi-214 609 in a NARROW window ---
    anchor_pts = list(coarse_pts)         # (channel, energy)
    anchor_w = [ANCHORS[n]['isolation'] for n in found]
    anchor_names = list(found.keys())
    ch_pred = energy_to_channel(ANCHORS['Bi-214']['energy'], coarse_cal)
    w = cfg['confirm_window_ch']
    res = _peak_centroid(counts, ch_pred - w, ch_pred + w, cfg)
    if res and res['significance'] >= cfg['min_significance']:
        anchor_pts.append((res['centroid'], ANCHORS['Bi-214']['energy']))
        anchor_w.append(ANCHORS['Bi-214']['isolation'])
        anchor_names.append('Bi-214')

    # --- Step 4: weighted final linear fit (weight = isolation reliability) ---
    chans = np.array([p[0] for p in anchor_pts])
    energies = np.array([p[1] for p in anchor_pts])
    weights = np.array(anchor_w)
    # polyfit returns highest-order first; linear -> [slope, intercept] in E(ch).
    slope, intercept = np.polyfit(chans, energies, 1, w=weights)
    final_cal = [float(intercept), float(slope), 0.0, 0.0]

    fitted = slope * chans + intercept
    rms = float(np.sqrt(np.mean((energies - fitted) ** 2)))

    # --- Step 5: geometric / sanity gate ---
    gain_ratio = slope / nominal_cal[1]
    if rms > cfg['max_rms_keV']:
        return None
    if not (cfg['min_gain_ratio'] <= gain_ratio <= cfg['max_gain_ratio']):
        return None

    # Apparent K-40 shift under the NOMINAL calibration (drift diagnostic).
    k40_shift = None
    if k40_centroid is not None:
        apparent = _energy_at(k40_centroid, nominal_cal)
        k40_shift = float(apparent - ANCHORS['K-40']['energy'])

    return {
        'calibration': final_cal,
        'rms_keV': rms,
        'n_anchors': len(anchor_pts),
        'anchors_used': anchor_names,
        'gain_ratio': float(gain_ratio),
        'k40_shift_keV': k40_shift,
    }


def correct_counts(counts, gain_calibration, nominal_cal):
    """Resample a drifted spectrum onto the nominal energy grid, conserving counts.

    counts[i] truly belongs at energy gain_calibration(i); this redistributes the
    counts so that the output channel j holds what belongs at nominal energy
    nominal_cal(j). Peaks therefore land back on their nominal channels. Total
    counts are preserved (cumulative-interpolation rebin).
    """
    counts = np.asarray(counts, dtype=float)
    n = len(counts)
    edges = np.arange(n + 1) - 0.5            # channel bin edges

    src_edge_E = _energy_at(edges, gain_calibration)   # true energy of each source edge
    tgt_edge_E = _energy_at(edges, nominal_cal)        # nominal energy of each target edge

    cum = np.concatenate([[0.0], np.cumsum(counts)])   # cumulative counts at source edges
    cum_at_tgt = np.interp(tgt_edge_E, src_edge_E, cum, left=0.0, right=cum[-1])
    corrected = np.diff(cum_at_tgt)
    corrected[corrected < 0] = 0.0
    return corrected
