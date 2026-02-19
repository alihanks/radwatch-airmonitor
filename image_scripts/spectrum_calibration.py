"""
Spectrum Calibration Module for Jupyter Notebooks

This module provides functions to calibrate gamma-ray spectra using natural 
background peaks. Designed for interactive use in Jupyter notebooks.

Common calibration peaks:
- Bi-214 (609 keV): Medium energy, usually strong
- K-40 (1461 keV): High energy, ubiquitous natural background
- Tl-208 (2615 keV): Very high energy, from Th-232 chain

Example usage in notebook:
    from spectrum_calibration import *
    
    # Load your spectrum
    counts = ...  # Your count data
    
    # Find peaks
    peaks = find_peaks_for_calibration(counts)
    
    # Select peaks for calibration (manually specify channels)
    calibration_points = [
        (peak_channel_1, 609.3),   # Bi-214
        (peak_channel_2, 1460.8),  # K-40
        (peak_channel_3, 2614.5),  # Tl-208
    ]
    
    # Calculate calibration
    cal = calibrate_spectrum(calibration_points, order=1)
    
    # Apply calibration
    energies = apply_calibration(counts, cal)
    
    # Plot results
    plot_calibrated_spectrum(counts, cal, calibration_points)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Reference peak energies (keV) for common natural background
REFERENCE_PEAKS = {
    'Bi-214 (609)': 609.3,
    'Bi-214 (1120)': 1120.3,
    'Bi-214 (1764)': 1764.5,
    'K-40': 1460.8,
    'Tl-208 (583)': 583.2,
    'Tl-208 (2615)': 2614.5,
    'Cs-137': 661.7,
    'Co-60 (1173)': 1173.2,
    'Co-60 (1332)': 1332.5,
}


def find_peaks_for_calibration(counts, min_height=10, min_prominence=5,
                               max_peaks=20, plot=True):
    """
    Find prominent peaks in spectrum for calibration.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    min_height : float
        Minimum peak height (counts)
    min_prominence : float
        Minimum peak prominence (counts)
    max_peaks : int
        Maximum number of peaks to return
    plot : bool
        If True, plot spectrum with identified peaks

    Returns
    -------
    peaks : ndarray
        Array of peak channel positions
    properties : dict
        Peak properties (heights, prominences, etc.)
    """
    peaks, properties = find_peaks(counts, height=min_height,
                                   prominence=min_prominence, distance=5)

    # Sort by height (most prominent first)
    sorted_indices = np.argsort(properties['peak_heights'])[::-1]
    peaks = peaks[sorted_indices][:max_peaks]

    # Get properties for sorted peaks
    peak_heights = properties['peak_heights'][sorted_indices][:max_peaks]
    peak_prominences = properties['prominences'][sorted_indices][:max_peaks]

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        channels = np.arange(len(counts))

        # Linear scale
        ax1.plot(channels, counts, 'b-', linewidth=0.5, alpha=0.7)
        ax1.plot(peaks, counts[peaks], 'r^', markersize=10, label='Found peaks')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Counts')
        ax1.set_title(f'Found {len(peaks)} peaks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Log scale
        ax2.semilogy(channels, counts + 1, 'b-', linewidth=0.5, alpha=0.7)
        ax2.semilogy(peaks, counts[peaks] + 1, 'r^', markersize=10, label='Found peaks')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Counts (log)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.show()

        # Print peak table
        print(f"\n{'#':<4} {'Channel':<10} {'Height':<12} {'Prominence':<12}")
        print('-' * 40)
        for i, (ch, h, p) in enumerate(zip(peaks, peak_heights, peak_prominences)):
            print(f"{i + 1:<4} {ch:<10} {h:<12.0f} {p:<12.1f}")

    return peaks, {'heights': peak_heights, 'prominences': peak_prominences}


def find_peaks_in_windows(counts, windows, plot=True):
    """
    Find the strongest peak in each specified window.

    This is more robust than global peak finding when you know approximately
    where peaks should be. Good for calibration with natural background peaks.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    windows : list of tuples
        List of (min_channel, max_channel) or (min_channel, max_channel, name) tuples.
        Each window should contain one expected peak.
        Example: [(280, 330, 'Bi-214'), (700, 760, 'K-40'), (1280, 1340, 'Tl-208')]
    plot : bool
        If True, plot spectrum with windows and found peaks

    Returns
    -------
    peak_results : list of dicts
        List of results for each window:
        {'window': (min, max), 'name': str, 'channel': int, 'counts': float}
    """
    peak_results = []

    for window in windows:
        if len(window) == 2:
            win_min, win_max = window
            name = f"Window {win_min}-{win_max}"
        else:
            win_min, win_max, name = window

        # Extract window region
        win_min = int(win_min)
        win_max = int(win_max)
        window_counts = counts[win_min:win_max + 1]

        if len(window_counts) == 0:
            print(f"Warning: Window {name} is empty or out of range")
            continue

        # Find peak in window (just the maximum)
        peak_idx_in_window = np.argmax(window_counts)
        peak_channel = win_min + peak_idx_in_window
        peak_counts = window_counts[peak_idx_in_window]

        peak_results.append({
            'window': (win_min, win_max),
            'name': name,
            'channel': peak_channel,
            'counts': peak_counts
        })

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        channels = np.arange(len(counts))

        # Linear scale
        ax1.plot(channels, counts, 'b-', linewidth=0.5, alpha=0.7, label='Spectrum')

        # Mark windows and peaks
        colors = plt.cm.Set1(np.linspace(0, 1, len(peak_results)))
        for i, result in enumerate(peak_results):
            win_min, win_max = result['window']
            peak_ch = result['channel']

            # Shade window
            ax1.axvspan(win_min, win_max, alpha=0.2, color=colors[i])

            # Mark peak
            ax1.plot(peak_ch, result['counts'], '^',
                     markersize=12, color=colors[i],
                     label=f"{result['name']} (ch {peak_ch})")

        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Counts')
        ax1.set_title(f'Peak Finding in {len(windows)} Windows')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Log scale
        ax2.semilogy(channels, counts + 1, 'b-', linewidth=0.5, alpha=0.7)

        for i, result in enumerate(peak_results):
            win_min, win_max = result['window']
            peak_ch = result['channel']

            ax2.axvspan(win_min, win_max, alpha=0.2, color=colors[i])
            ax2.plot(peak_ch, result['counts'] + 1, '^',
                     markersize=12, color=colors[i])

        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Counts (log)')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.show()

    # Print results table
    print(f"\n{'Window':<25} {'Channel':<10} {'Counts':<12}")
    print('-' * 50)
    for result in peak_results:
        win_min, win_max = result['window']
        print(f"{result['name']:<25} {result['channel']:<10} {result['counts']:<12.0f}")

    return peak_results


def calibrate_spectrum(channel_energy_pairs, order=1):
    """
    Calculate calibration coefficients from channel-energy pairs.

    Parameters
    ----------
    channel_energy_pairs : list of tuples
        List of (channel, energy_keV) pairs
        Example: [(123, 609.3), (456, 1460.8), (789, 2614.5)]
    order : int
        Polynomial order (1=linear, 2=quadratic)

    Returns
    -------
    calibration : list
        Calibration coefficients [c0, c1, c2, c3]
        Energy = c0 + c1*ch + c2*ch^2 + c3*ch^3
    fit_info : dict
        Information about fit quality
    """
    if len(channel_energy_pairs) < order + 1:
        raise ValueError(f"Need at least {order + 1} points for order {order} fit. "
                         f"Only got {len(channel_energy_pairs)} points.")

    channels = np.array([pair[0] for pair in channel_energy_pairs])
    energies = np.array([pair[1] for pair in channel_energy_pairs])

    # Fit polynomial: Energy = p[0]*ch^n + p[1]*ch^(n-1) + ... + p[n]
    coeffs = np.polyfit(channels, energies, order)

    # Convert to our format [c0, c1, c2, c3] where Energy = c0 + c1*ch + c2*ch^2 + c3*ch^3
    calibration = [0.0, 0.0, 0.0, 0.0]
    for i, c in enumerate(reversed(coeffs)):
        calibration[i] = c

    # Calculate fit quality
    fitted_energies = np.polyval(coeffs, channels)
    residuals = energies - fitted_energies
    rms_error = np.sqrt(np.mean(residuals**2))

    fit_info = {
        'rms_error': rms_error,
        'residuals': residuals,
        'channels': channels,
        'energies': energies,
        'fitted_energies': fitted_energies,
        'order': order
    }

    return calibration, fit_info


def apply_calibration(counts, calibration):
    """
    Apply calibration to get energy axis.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    calibration : list
        Calibration coefficients [c0, c1, c2, c3]

    Returns
    -------
    energies : ndarray
        Energy values for each channel (keV)
    """
    channels = np.arange(len(counts))
    c0, c1, c2, c3 = calibration
    energies = c0 + c1 * channels + c2 * channels**2 + c3 * channels**3
    return energies


def plot_calibration_fit(fit_info):
    """
    Plot calibration fit showing residuals.

    Parameters
    ----------
    fit_info : dict
        Output from calibrate_spectrum()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Fit plot
    ax1.plot(fit_info['channels'], fit_info['energies'], 'ro',
             markersize=10, label='Calibration points')

    # Dense channel range for smooth line
    ch_range = np.linspace(fit_info['channels'].min(),
                           fit_info['channels'].max(), 1000)
    if fit_info['order'] == 1:
        # Linear: E = c0 + c1*ch
        fit_line = fit_info['fitted_energies'][0] + \
            (fit_info['fitted_energies'][1] - fit_info['fitted_energies'][0]) / \
            (fit_info['channels'][1] - fit_info['channels'][0]) * \
            (ch_range - fit_info['channels'][0])
    else:
        # Use polyfit coefficients
        coeffs = np.polyfit(fit_info['channels'], fit_info['energies'],
                            fit_info['order'])
        fit_line = np.polyval(coeffs, ch_range)

    ax1.plot(ch_range, fit_line, 'b-', linewidth=2, label='Fit')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Energy (keV)')
    ax1.set_title(f'Calibration Fit (Order {fit_info["order"]}, '
                  f'RMS error: {fit_info["rms_error"]:.2f} keV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    ax2.plot(fit_info['channels'], fit_info['residuals'], 'ro-', markersize=8)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.axhline(fit_info['rms_error'], color='r', linestyle=':',
                linewidth=1, label=f'±RMS ({fit_info["rms_error"]:.2f} keV)')
    ax2.axhline(-fit_info['rms_error'], color='r', linestyle=':', linewidth=1)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Residual (keV)')
    ax2.set_title('Calibration Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print(f"\n{'Peak #':<8} {'Channel':<10} {'Expected':<12} {'Fitted':<12} {'Error':<10}")
    print('-' * 52)
    for i, (ch, exp, fit) in enumerate(zip(fit_info['channels'],
                                           fit_info['energies'],
                                           fit_info['fitted_energies'])):
        error = fit - exp
        print(f"{i + 1:<8} {ch:<10.0f} {exp:<12.1f} {fit:<12.1f} {error:>+10.2f}")


def plot_calibrated_spectrum(counts, calibration, calibration_points=None,
                             xlim=(0, 3000), figsize=(14, 10)):
    """
    Plot calibrated spectrum with marked calibration peaks.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    calibration : list
        Calibration coefficients [c0, c1, c2, c3]
    calibration_points : list of tuples, optional
        List of (channel, energy) pairs used for calibration
    xlim : tuple
        Energy range to display (keV)
    figsize : tuple
        Figure size
    """
    energies = apply_calibration(counts, calibration)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Linear scale
    ax1.plot(energies, counts, 'b-', linewidth=0.8)
    ax1.set_xlabel('Energy (keV)', fontsize=12)
    ax1.set_ylabel('Counts', fontsize=12)

    # Format calibration equation for title
    c0, c1, c2, c3 = calibration
    title = f'Calibrated Spectrum\nE = {c0:.3f} + {c1:.6f}×ch'
    if abs(c2) > 1e-10:
        title += f' + {c2:.9f}×ch²'
    if abs(c3) > 1e-15:
        title += f' + {c3:.12f}×ch³'
    ax1.set_title(title, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)

    # Mark calibration peaks
    if calibration_points:
        for channel, expected_energy in calibration_points:
            if xlim[0] <= expected_energy <= xlim[1]:
                ax1.axvline(expected_energy, color='r', linestyle='--',
                            alpha=0.5, linewidth=1.5)
                ax1.text(expected_energy, ax1.get_ylim()[1] * 0.95,
                         f'{expected_energy:.0f} keV',
                         rotation=90, va='top', ha='right', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', alpha=0.7))

    # Log scale
    ax2.semilogy(energies, counts + 1, 'b-', linewidth=0.8)
    ax2.set_xlabel('Energy (keV)', fontsize=12)
    ax2.set_ylabel('Counts (log)', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(xlim)

    if calibration_points:
        for channel, expected_energy in calibration_points:
            if xlim[0] <= expected_energy <= xlim[1]:
                ax2.axvline(expected_energy, color='r', linestyle='--',
                            alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    return fig


def print_calibration_summary(calibration, fit_info=None):
    """
    Print a summary of the calibration.

    Parameters
    ----------
    calibration : list
        Calibration coefficients [c0, c1, c2, c3]
    fit_info : dict, optional
        Fit information from calibrate_spectrum()
    """
    c0, c1, c2, c3 = calibration

    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print("\nCoefficients:")
    print(f"  c0 = {c0:.6f}")
    print(f"  c1 = {c1:.9f}")
    print(f"  c2 = {c2:.12f}")
    print(f"  c3 = {c3:.15f}")

    print("\nCalibration equation:")
    print(f"  Energy (keV) = {c0:.6f} + {c1:.9f}×ch", end='')
    if abs(c2) > 1e-10:
        print(f" + {c2:.12f}×ch²", end='')
    if abs(c3) > 1e-15:
        print(f" + {c3:.15f}×ch³", end='')
    print()

    if fit_info:
        print(f"\nFit quality:")
        print(f"  RMS error: {fit_info['rms_error']:.3f} keV")
        print(f"  Calibration points: {len(fit_info['channels'])}")
        print(f"  Polynomial order: {fit_info['order']}")

    print("\nTo use in code:")
    print(f"  calibration = {calibration}")
    print(f"  energies = apply_calibration(counts, calibration)")
    print("=" * 70)


def suggest_calibration_peaks(counts, expected_peaks=None):
    """
    Suggest which peaks in the spectrum might correspond to known energies.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    expected_peaks : dict, optional
        Dictionary of {name: energy_keV}. If None, uses REFERENCE_PEAKS.

    Returns
    -------
    suggestions : list of tuples
        List of (channel, suggested_energy, peak_name, confidence)
    """
    if expected_peaks is None:
        expected_peaks = {
            'Bi-214 (609)': 609.3,
            'K-40': 1460.8,
            'Tl-208 (2615)': 2614.5,
        }

    # Find peaks
    peaks, props = find_peaks_for_calibration(counts, plot=False)

    # Very rough guess at calibration assuming ~0.5-2 keV/channel
    # This is just to help identify peaks - not for actual calibration
    keV_per_ch_guesses = [0.5, 0.75, 1.0, 1.5, 2.0]

    suggestions = []

    print("\nPossible peak identifications:")
    print("=" * 70)
    print(f"{'Channel':<10} {'Counts':<12} {'Possible Identity':<30} {'Approx Energy'}")
    print("-" * 70)

    for peak_ch in peaks[:10]:  # Check top 10 peaks
        peak_counts = counts[peak_ch]

        # Try different calibration guesses
        for keV_per_ch in keV_per_ch_guesses:
            approx_energy = peak_ch * keV_per_ch

            # See if it matches any expected peak (within 20%)
            for peak_name, expected_energy in expected_peaks.items():
                if abs(approx_energy - expected_energy) / expected_energy < 0.2:
                    print(f"{peak_ch:<10} {peak_counts:<12.0f} {peak_name:<30} "
                          f"~{approx_energy:.0f} keV (if {keV_per_ch} keV/ch)")

    print("\nTo use these suggestions:")
    print("  1. Examine the plotted spectrum to identify strong peaks")
    print("  2. Match peak channels to expected energies")
    print("  3. Create calibration_points = [(ch1, E1), (ch2, E2), ...]")
    print("  4. Run: cal, info = calibrate_spectrum(calibration_points)")


# Convenience function for quick calibration workflow
def quick_calibrate(counts, calibration_points, order=1, plot=True):
    """
    Quick calibration workflow: calibrate and plot in one step.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    calibration_points : list of tuples
        List of (channel, energy_keV) pairs
    order : int
        Polynomial order (1=linear, 2=quadratic)
    plot : bool
        If True, show plots

    Returns
    -------
    calibration : list
        Calibration coefficients
    energies : ndarray
        Calibrated energy axis
    fit_info : dict
        Fit information
    """
    # Calculate calibration
    calibration, fit_info = calibrate_spectrum(calibration_points, order=order)

    # Apply calibration
    energies = apply_calibration(counts, calibration)

    # Print summary
    print_calibration_summary(calibration, fit_info)

    if plot:
        # Plot fit quality
        plot_calibration_fit(fit_info)

        # Plot calibrated spectrum
        plot_calibrated_spectrum(counts, calibration, calibration_points)

    return calibration, energies, fit_info


def calibrate_from_windows(counts, windows, expected_energies, order=1, plot=True):
    """
    Complete calibration workflow using channel windows.

    This is the easiest way to calibrate when you know approximately where
    peaks should be. Just specify windows and expected energies, and this
    function does everything.

    Parameters
    ----------
    counts : array-like
        Spectrum count data
    windows : list of tuples
        List of (min_channel, max_channel) or (min_channel, max_channel, name).
        Example: [(280, 330, 'Bi-214'), (700, 760, 'K-40')]
    expected_energies : list of floats
        Expected energy for each window (keV), in same order as windows.
        Example: [609.3, 1460.8]
    order : int
        Polynomial order (1=linear, 2=quadratic)
    plot : bool
        If True, show plots

    Returns
    -------
    calibration : list
        Calibration coefficients [c0, c1, c2, c3]
    energies : ndarray
        Calibrated energy axis
    fit_info : dict
        Fit information
    peak_results : list of dicts
        Peak finding results for each window

    Examples
    --------
    >>> # Define windows where you expect peaks
    >>> windows = [
    ...     (280, 330, 'Bi-214'),   # Bi-214 609 keV
    ...     (700, 760, 'K-40'),      # K-40 1461 keV
    ...     (1280, 1340, 'Tl-208')   # Tl-208 2615 keV
    ... ]
    >>> expected_energies = [609.3, 1460.8, 2614.5]
    >>> 
    >>> # Do everything in one step
    >>> cal, energies, info, peaks = calibrate_from_windows(
    ...     counts, windows, expected_energies, order=1
    ... )
    """
    if len(windows) != len(expected_energies):
        raise ValueError(f"Number of windows ({len(windows)}) must match "
                         f"number of expected energies ({len(expected_energies)})")

    if len(windows) < order + 1:
        raise ValueError(f"Need at least {order + 1} peaks for order {order} fit. "
                         f"Only got {len(windows)} windows.")

    # Find peaks in windows
    print("=" * 70)
    print("FINDING PEAKS IN WINDOWS")
    print("=" * 70)
    peak_results = find_peaks_in_windows(counts, windows, plot=plot)

    # Create calibration points
    calibration_points = []
    for result, energy in zip(peak_results, expected_energies):
        calibration_points.append((result['channel'], energy))

    print(f"\n{'=' * 70}")
    print("CALCULATING CALIBRATION")
    print(f"{'=' * 70}")

    # Calculate calibration
    calibration, fit_info = calibrate_spectrum(calibration_points, order=order)

    # Apply calibration
    energies = apply_calibration(counts, calibration)

    # Print summary
    print_calibration_summary(calibration, fit_info)

    if plot:
        # Plot fit quality
        plot_calibration_fit(fit_info)

        # Plot calibrated spectrum
        plot_calibrated_spectrum(counts, calibration, calibration_points)

    return calibration, energies, fit_info, peak_results


def energy_to_channel(energy_keV, calibration):
    """Convert energy (keV) to channel number.
    Inverts E = c0 + c1*ch  =>  ch = (E - c0) / c1
    Only valid for linear calibration (c2=c3=0)."""
    c0, c1, c2, c3 = calibration
    return (energy_keV - c0) / c1


def read_calibration_file(filepath):
    """
    Read calibration coefficients from saved text file.

    Reads the calibration coefficients file created by saving results
    in the notebook examples.

    Parameters
    ----------
    filepath : str
        Path to calibration coefficients text file

    Returns
    -------
    calibration : list
        Calibration coefficients [c0, c1, c2, c3]

    Examples
    --------
    >>> cal = read_calibration_file('calibration_coefficients.txt')
    >>> energies = apply_calibration(counts, cal)
    """
    calibration = [0.0, 0.0, 0.0, 0.0]

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c0 ='):
                calibration[0] = float(line.split('=')[1].strip())
            elif line.startswith('c1 ='):
                calibration[1] = float(line.split('=')[1].strip())
            elif line.startswith('c2 ='):
                calibration[2] = float(line.split('=')[1].strip())
            elif line.startswith('c3 ='):
                calibration[3] = float(line.split('=')[1].strip())

    # Verify we got at least c0 and c1
    if calibration[0] == 0.0 and calibration[1] == 0.0:
        raise ValueError(f"Could not read calibration coefficients from {filepath}")

    return calibration
