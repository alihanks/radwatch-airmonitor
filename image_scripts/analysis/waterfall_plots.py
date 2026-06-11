"""waterfall_plots.py — daily spectral waterfall + gain-trend plots.

Reads the last 24 h of raw 300 s CNFs straight from the Dropbox `current/`
year folders (rebin.h5 only stores hourly sums, which are too coarse for
these views) and produces:

  data/waterfall_energy.png             website-facing: energy-axis waterfall
  data/waterfalls/current_channel.png   diagnostic: raw channel-axis waterfall
  data/waterfalls/current_gain_trend.png diagnostic: calibration coeffs vs time

plus per-calendar-day archives (data/waterfalls/YYYY-MM-DD_*.png), written once
the first time the script runs after that day has ended.

The data/waterfalls/ subfolder is deliberately NOT deployed: cron_job.sh only
copies top-level data/*.png to the website staging dir, so the diagnostics stay
server-side while waterfall_energy.png ships.

Gain policy — "stable unless proven otherwise": the detector is supposed to be
gain-stable, so per-spectrum gain fits (gain_stabilization) are always run for
DETECTION, but the energy axis only uses a fitted calibration when a real shift
is detected (|gain_ratio - 1| > GAIN_SHIFT_THRESHOLD). Otherwise — including
when no fit is possible — the nominal calibration is applied, so stable data is
never resampled by fit noise (±0.3% on a stable day vs 3-9% for real shifts),
and collapse-mode spectra show up honestly as the collapse rather than being
"corrected".

Run by cron_job.sh after h5_analysis.py; safe to run standalone:
    python3 image_scripts/analysis/waterfall_plots.py
"""

import sys
import os
import re
import glob
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/')
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/image_scripts')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from image_scripts import cnf_parser_standalone as cnf_parser
from image_scripts import gain_stabilization as gs
from image_scripts.spectrum_calibration import read_calibration_file

PROJECT_ROOT = '/home/dosenet/radwatch-airmonitor'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
WATERFALL_DIR = os.path.join(DATA_DIR, 'waterfalls')
SPEC_DIR = r'/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/'

# Apply a fitted calibration only beyond this |gain_ratio - 1|. Stable-day fit
# noise is ~±0.003; real shifts observed so far are 0.03-0.09.
GAIN_SHIFT_THRESHOLD = 0.01

# Reference lines marked on the energy waterfall (keV).
MARK_LINES = [(351.9, 'Pb-214 352'), (609.3, 'Bi-214 609'),
              (1460.8, 'K-40 1461'), (2614.5, 'Tl-208 2615')]

FNAME_TS_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})')


def cnf_timestamp(path):
    """Acquisition timestamp from the CNF filename, or None."""
    m = FNAME_TS_RE.search(os.path.basename(path))
    return datetime.datetime(*map(int, m.groups())) if m else None


def collect_cnfs(start_dt, end_dt):
    """All CNFs in the Dropbox year folders with start_dt <= timestamp < end_dt."""
    out = []
    for path in glob.glob(os.path.join(SPEC_DIR, '[0-9][0-9][0-9][0-9]', '*.CNF')):
        ts = cnf_timestamp(path)
        if ts is not None and start_dt <= ts < end_dt:
            out.append((ts, path))
    out.sort()
    return out


def load_spectra(entries):
    """Parse CNFs -> (times, matrix). Skips unreadable files."""
    times, rows = [], []
    for ts, path in entries:
        try:
            counts = np.asarray(cnf_parser.read_cnf(path)['counts'], dtype=float)
        except Exception as e:
            print(f"  skipping unreadable CNF {os.path.basename(path)}: {e}")
            continue
        times.append(ts)
        rows.append(counts)
    if not rows:
        return [], None
    n_ch = max(len(r) for r in rows)
    M = np.zeros((len(rows), n_ch))
    for i, r in enumerate(rows):
        M[i, :len(r)] = r
    return times, M


def plot_channel_waterfall(times, M, out_path, label):
    fac = max(1, M.shape[1] // 512)
    nb = M.shape[1] // fac
    Mr = M[:, :nb * fac].reshape(M.shape[0], nb, fac).sum(2)
    centers = np.arange(nb) * fac + fac / 2.0
    fig, ax = plt.subplots(figsize=(9, 9))
    pcm = ax.pcolormesh(centers, times, Mr, norm=LogNorm(vmin=1, vmax=max(Mr.max(), 2)),
                        cmap='turbo', shading='nearest')
    ax.set_xlabel('Channel (raw, uncalibrated)')
    ax.set_ylabel('Time (local)')
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.set_title(f'Spectral waterfall, CHANNEL space (diagnostic) — {label}')
    fig.colorbar(pcm, ax=ax, label='counts/bin (log)')
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_energy_waterfall(times, M, results, nominal_cal, out_path, label):
    """Energy-axis waterfall. Nominal calibration everywhere, except spectra
    whose fitted gain deviates more than GAIN_SHIFT_THRESHOLD (those use their
    fitted calibration). Returns the number of gain-corrected spectra."""
    n_ch = M.shape[1]
    b_nom = nominal_cal[1] * gs.DEFAULT_CONFIG['ref_channels'] / n_ch
    nominal = [nominal_cal[0], b_nom, 0.0, 0.0]
    ch = np.arange(n_ch)
    edges = np.arange(0, 2820, 10.0)
    ecent = (edges[:-1] + edges[1:]) / 2
    Me = np.zeros((M.shape[0], len(ecent)))
    n_corrected = 0
    for i in range(M.shape[0]):
        r = results[i]
        if r is not None and abs(r['gain_ratio'] - 1.0) > GAIN_SHIFT_THRESHOLD:
            a, b = r['calibration'][0], r['calibration'][1]
            n_corrected += 1
        else:
            a, b = nominal[0], nominal[1]
        Me[i], _ = np.histogram(a + b * ch, bins=edges, weights=M[i])
    fig, ax = plt.subplots(figsize=(9, 9))
    pcm = ax.pcolormesh(ecent, times, Me, norm=LogNorm(vmin=1, vmax=max(Me.max(), 2)),
                        cmap='turbo', shading='nearest')
    for e, lab in MARK_LINES:
        ax.axvline(e, color='w', ls='--', lw=0.7, alpha=0.6)
        ax.text(e, times[0], ' ' + lab, color='w', fontsize=8, rotation=90,
                va='top', ha='right')
    ax.set_xlim(0, 2800)
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Time (local)')
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    title = f'Gamma spectral waterfall — {label}'
    if n_corrected:
        title += f'  [{n_corrected} gain-corrected]'
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label='counts/10 keV (log)')
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return n_corrected


def plot_gain_trend(times, results, M_nch, nominal_cal, out_path, label):
    b_nom = nominal_cal[1] * gs.DEFAULT_CONFIG['ref_channels'] / M_nch
    t_ok = [t for t, r in zip(times, results) if r]
    g_ok = [r['gain_ratio'] for r in results if r]
    t_bad = [t for t, r in zip(times, results) if r is None]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t_ok, g_ok, '.', ms=4, color='#2D637F', label='gain ratio (fit/nominal)')
    ax.axhline(1.0, color='k', ls='--', lw=1, label=f'nominal (b={b_nom:.4f} keV/ch)')
    ax.axhspan(1 - GAIN_SHIFT_THRESHOLD, 1 + GAIN_SHIFT_THRESHOLD,
               color='g', alpha=0.08, label=f'±{100*GAIN_SHIFT_THRESHOLD:.0f}% (no correction)')
    for t in t_bad:
        ax.axvline(t, color='r', alpha=0.3, lw=1.2)
    if t_bad:
        ax.plot([], [], color='r', alpha=0.5, lw=1.2, label=f'no fit ({len(t_bad)})')
    ax.set_ylabel('gain ratio')
    ax.set_xlabel('Time (local)')
    ax.set_title(f'Detector gain vs time (diagnostic) — {label}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def generate_window(start_dt, end_dt, nominal_cal, label, energy_path,
                    channel_path, trend_path):
    """Generate all three plots for one time window. Returns True if produced."""
    entries = collect_cnfs(start_dt, end_dt)
    if len(entries) < 3:
        print(f"  {label}: only {len(entries)} CNFs in window, skipping plots")
        return False
    times, M = load_spectra(entries)
    if M is None or len(times) < 3:
        print(f"  {label}: too few readable spectra, skipping plots")
        return False
    print(f"  {label}: {len(times)} spectra "
          f"({times[0]:%Y-%m-%d %H:%M} .. {times[-1]:%H:%M}), {M.shape[1]} channels")
    results = gs.fit_collection_gains(list(M), nominal_cal)
    n_fit = sum(1 for r in results if r)
    plot_channel_waterfall(times, M, channel_path, label)
    n_corr = plot_energy_waterfall(times, M, results, nominal_cal, energy_path, label)
    plot_gain_trend(times, results, M.shape[1], nominal_cal, trend_path, label)
    print(f"  {label}: {n_fit}/{len(results)} gain fits, "
          f"{n_corr} spectra above ±{100*GAIN_SHIFT_THRESHOLD:.0f}% -> corrected")
    return True


def main():
    os.makedirs(WATERFALL_DIR, exist_ok=True)
    nominal_cal = read_calibration_file(os.path.join(
        PROJECT_ROOT, 'image_scripts', 'calibration', 'calibration_coefficients.txt'))
    now = datetime.datetime.now()

    # Rolling last-24h plots (regenerated every run).
    print("Generating last-24h waterfalls...")
    generate_window(
        now - datetime.timedelta(hours=24), now, nominal_cal, 'last 24 h',
        energy_path=os.path.join(DATA_DIR, 'waterfall_energy.png'),
        channel_path=os.path.join(WATERFALL_DIR, 'current_channel.png'),
        trend_path=os.path.join(WATERFALL_DIR, 'current_gain_trend.png'))

    # Per-calendar-day archive: first run after a day ends writes that day once.
    yesterday = (now - datetime.timedelta(days=1)).date()
    tag = yesterday.isoformat()
    archive_energy = os.path.join(WATERFALL_DIR, f'{tag}_energy.png')
    if not os.path.exists(archive_energy):
        print(f"Archiving waterfalls for {tag}...")
        day_start = datetime.datetime.combine(yesterday, datetime.time.min)
        generate_window(
            day_start, day_start + datetime.timedelta(days=1), nominal_cal, tag,
            energy_path=archive_energy,
            channel_path=os.path.join(WATERFALL_DIR, f'{tag}_channel.png'),
            trend_path=os.path.join(WATERFALL_DIR, f'{tag}_gain_trend.png'))
    else:
        print(f"Archive for {tag} already exists, skipping")


if __name__ == '__main__':
    main()
