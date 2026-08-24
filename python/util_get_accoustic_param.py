#!/usr/bin/env python3
"""
util_get_accoustic_param.py
==============================
Estimates the noise/reverb/calibration parameters used throughout this
project's simulation pipeline -- directly from a real recording, rather
than by hand-tuning. This is the estimation code that SHOULD have produced
gen_config.ini's [calibration] constants (they're documented as "measured from
glock3_ch0.wav" but no actual measurement code existed until this script).

What's estimable from a single passive recording (no special calibration
signal needed):
  - Noise floor RMS + its PSD slope (color)          -> [calibration]/[noise]
  - RT60 (via Schroeder backward energy integration)  -> [calibration]/[noise]
  - Peak amplitude (normalized)                       -> [calibration]
  - Approximate mic passband corners (see caveat below) -> [adc]

What this CANNOT give you, and why:
  - The exact room impulse response (only its RT60 summary statistic) --
    getting the real IR needs an active calibration signal (a sweep/chirp
    played through the same space), not just a passive recording.
  - True absolute Pa calibration -- without a known reference source level,
    only RELATIVE amplitude (normalized units) is recoverable.
  - Mic frequency response cleanly separated from room/environment coloring
    -- both show up together in the noise floor's spectrum; the passband
    estimate below is a mixture of both, not a clean mic-only measurement.

Usage
-----
  python util_get_accoustic_param.py recording.wav
  python util_get_accoustic_param.py recording.wav --config-out estimated.ini
"""

import argparse
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, correlate


# ---------------------------------------------------------------------------
# Self-contained: these were factored out of a separate similarity-analysis
# module so this script has no external project dependency beyond numpy/scipy.
# ---------------------------------------------------------------------------
def load_wav_normalized(path):
    """Load a .wav, take channel 0 if multichannel, normalize to peak = 1."""
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    return fs, x


def pearson_correlation_coefficient(x, y):
    """Standard Pearson correlation coefficient at zero lag (no shifting).
    Requires x and y to be the same length."""
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.std(x) * np.std(y) * len(x))
    if denom < 1e-20:
        return np.nan
    return float(np.sum(x * y) / denom)


def cross_correlation_coefficient(x1, x2, fs, max_lag_s=None):
    """Cross-correlation coefficient (mean-subtracted, std-normalized) as a
    function of lag; returns the value AND lag at the best-aligning point.
    This is the textbook cross-correlation coefficient function rho(tau),
    reduces to pearson_correlation_coefficient() at tau=0."""
    a = x1 - x1.mean()
    b = x2 - x2.mean()
    norm = np.std(a) * np.std(b) * min(len(a), len(b))
    if norm < 1e-20:
        return np.nan, 0.0
    corr = correlate(a, b, mode="full") / norm
    lags = np.arange(-len(b) + 1, len(a))
    if max_lag_s is not None:
        max_lag_n = int(max_lag_s * fs)
        mask = np.abs(lags) <= max_lag_n
        corr, lags = corr[mask], lags[mask]
    idx = np.argmax(np.abs(corr))
    # NOTE: scipy.signal.correlate's native lag sign is the OPPOSITE of the
    # intuitive "positive lag = x2 arrives after x1" convention -- verified
    # empirically with a two-impulse test case. Negate it here so the
    # reported lag means what it says.
    return float(corr[idx]), -lags[idx] / fs


# ---------------------------------------------------------------------------
# Frequency-domain similarity
# ---------------------------------------------------------------------------


def psd_welch(x, fs, nperseg=2048):
    return welch(x, fs=fs, nperseg=min(nperseg, len(x)))


def psd_slope_fit(f, Pxx, f_lo=100, f_hi=10000):
    """Fit dB/decade slope of the PSD over [f_lo, f_hi] Hz."""
    mask = (f >= f_lo) & (f <= f_hi) & (Pxx > 0)
    if mask.sum() < 3:
        return np.nan
    logf, logP = np.log10(f[mask]), 10 * np.log10(Pxx[mask])
    slope, _ = np.polyfit(logf, logP, 1)
    return slope


def schroeder_edc_db(x, fs):
    """Backward-integrated (Schroeder 1965) energy decay curve, in dB."""
    edc = np.cumsum((x[::-1].astype(np.float64)) ** 2)[::-1]
    edc = edc / (edc[0] + 1e-20)
    return np.arange(len(x)) / fs, 10 * np.log10(edc + 1e-20)


def estimate_rt60_schroeder(x, fs, fit_range_db=(-5, -25)):
    t, edc_db = schroeder_edc_db(x, fs)
    hi, lo = fit_range_db
    mask = (edc_db <= hi) & (edc_db >= lo)
    if mask.sum() < 10:
        return np.nan
    slope, _ = np.polyfit(t[mask], edc_db[mask], 1)
    return -60.0 / slope if slope < 0 else np.nan


# ---------------------------------------------------------------------------
# Onset / pulse-shape proxy
# ---------------------------------------------------------------------------


def find_loudest_onset(x, fs, thresh_mult=8.0, noise_window_s=0.005):
    n = max(1, int(noise_window_s * fs))
    noise_rms = np.sqrt(np.mean(x[:n] ** 2))
    idx = np.where(np.abs(x) > thresh_mult * noise_rms)[0]
    return (int(idx[0]), noise_rms) if len(idx) else (None, noise_rms)




def check_channel_quality(wav_paths, max_lag_s=0.01, flag_threshold=0.85):
    """Cross-check multiple single-channel recordings of the SAME event (as
    this project's datasets are stored -- one .wav per mic) against their
    own consensus mean, to catch problem channels (wiring faults, polarity
    inversion, bad connections) before they contaminate a calibration
    estimate. max_lag_s allows for the small, LEGITIMATE timing differences
    between physically separated mics (that's real TDOA information, not a
    fault) -- a channel that still correlates poorly even within that
    allowance is a genuine data-quality problem, not just array geometry."""
    fs_list, signals = [], []
    for p in wav_paths:
        fs, x = load_wav_normalized(p)
        fs_list.append(fs)
        signals.append(x)
    if len(set(fs_list)) > 1:
        print("WARNING: channels have different sample rates -- comparison may be unreliable.",
              file=sys.stderr)
    fs = fs_list[0]
    n = min(len(s) for s in signals)
    mean_sig = np.mean([s[:n] for s in signals], axis=0)

    print(f"\n{'Channel':<24} {'r (0-lag)':>12} {'r (best-lag)':>14} {'best lag (ms)':>14}   Flag")
    print("-" * 78)
    results = []
    for path, x in zip(wav_paths, signals):
        r0 = pearson_correlation_coefficient(x[:n], mean_sig)
        r_best, lag_s = cross_correlation_coefficient(x, mean_sig, fs, max_lag_s=max_lag_s)
        flag = ""
        if r_best < 0:
            flag = "INVERTED POLARITY"
        elif r_best < flag_threshold:
            flag = "LOW CORRELATION"
        results.append(dict(path=path, r0=r0, r_best=r_best, lag_s=lag_s, flag=flag))
        print(f"{path:<24} {r0:>12.4f} {r_best:>14.4f} {lag_s*1000:>14.4f}   {flag}")

    good = [r for r in results if not r["flag"]]
    if good:
        best = max(good, key=lambda r: r["r_best"])
        print(f"\nRecommended channel (highest best-lag correlation, no flags): {best['path']}")
    else:
        print("\nWARNING: every channel was flagged -- inspect manually before trusting any of them.")
    return results


def estimate_rt60_robust(tail, fs, candidate_ranges=((-5,-15), (-5,-25), (-5,-35))):
    """RT60 via Schroeder backward integration, trying several standard fit
    windows (T10/T20/T30-style) and reporting fit quality (R^2) for each --
    a single fixed window can silently give a bogus answer if the real decay
    isn't a clean single exponential there (common in real rooms: an early
    fast drop, then a slower low-level tail before the true noise floor).
    Returns the best-R^2 estimate plus all candidates for transparency."""
    t, edc_db = schroeder_edc_db(tail, fs)
    results = []
    for hi, lo in candidate_ranges:
        mask = (edc_db <= hi) & (edc_db >= lo)
        if mask.sum() < 10:
            results.append(dict(range=(hi, lo), rt60=float("nan"), r2=float("nan"), n_points=int(mask.sum())))
            continue
        slope, intercept = np.polyfit(t[mask], edc_db[mask], 1)
        pred = slope * t[mask] + intercept
        ss_res = np.sum((edc_db[mask] - pred) ** 2)
        ss_tot = np.sum((edc_db[mask] - edc_db[mask].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rt60 = -60.0 / slope if slope < 0 else float("nan")
        results.append(dict(range=(hi, lo), rt60=rt60, r2=float(r2), n_points=int(mask.sum())))

    valid = [r for r in results if np.isfinite(r["r2"])]
    best = max(valid, key=lambda r: r["r2"]) if valid else results[0]
    return best, results


def estimate_noise_floor(x, fs, onset_idx, pre_pad_s=0.002):
    """RMS + PSD slope of the quiet region BEFORE the detected event.
    NOTE 1: psd_slope_fit returns dB/decade (10*log10(Pxx) vs log10(f)); the
    project's noise model (gunshot_physics.make_colored_noise) instead
    expects the PSD POWER-LAW EXPONENT directly (white=0, pink=-1, brown=-2,
    via `freq ** (slope/2)` on the amplitude spectrum) -- these differ by a
    factor of 10, an easy-to-miss unit bug that's fixed here (see the /10.0
    below).
    NOTE 2: like the mic passband estimate, this slope reflects mic/ADC
    response MIXED with true environmental noise color -- a passive
    recording alone can't cleanly separate the two."""
    end = max(1, (onset_idx or len(x)) - int(pre_pad_s * fs))
    quiet = x[:end]
    if len(quiet) < int(0.005 * fs):
        print("WARNING: quiet pre-event region is very short (<5ms) -- "
              "noise floor estimate may be unreliable.", file=sys.stderr)
    noise_rms = float(np.sqrt(np.mean(quiet ** 2))) if len(quiet) else float("nan")
    f, Pxx = psd_welch(quiet, fs, nperseg=min(2048, max(64, len(quiet))))
    slope_db_per_decade = psd_slope_fit(f, Pxx, f_lo=max(20, f[1]), f_hi=fs / 2 * 0.9)
    slope_exponent = slope_db_per_decade / 10.0   # <-- unit conversion for gunshot_physics.py
    return noise_rms, slope_exponent, quiet, f, Pxx


def estimate_mic_passband(f, Pxx, ref_band=(200, 5000), drop_db=3.0):
    """Rough passband corner estimate from the noise floor's own spectrum.
    CAVEAT: this mixes true mic/ADC frequency response with whatever the
    environment's own noise coloring looks like -- it is NOT a clean mic-
    only measurement. Treat as an approximate starting point, not ground
    truth; a proper measurement needs a calibration sweep through the same
    signal chain (silence in, known sweep in, compare)."""
    logP = 10 * np.log10(Pxx + 1e-20)
    ref_mask = (f >= ref_band[0]) & (f <= ref_band[1])
    if not ref_mask.any():
        return float(f[1]), float(f[-1])
    ref_level = np.median(logP[ref_mask])
    threshold = ref_level - drop_db

    above = logP >= threshold
    idx = np.where(above)[0]
    if len(idx) == 0:
        return float(f[1]), float(f[-1])
    lo = float(f[max(idx[0], 1)])
    hi = float(f[idx[-1]])
    return lo, hi


def estimate_all(wav_path, event_thresh_mult=8.0):
    fs, x = load_wav_normalized(wav_path)
    print(f"{wav_path}: fs={fs} Hz, duration={len(x)/fs:.3f} s, dtype from file")

    onset_idx, _ = find_loudest_onset(x, fs, thresh_mult=event_thresh_mult)
    if onset_idx is None:
        print("WARNING: no clear onset found above threshold -- treating whole "
              "file as 'quiet' for noise floor estimation, and skipping RT60.",
              file=sys.stderr)
        noise_rms, slope, quiet, f, Pxx = estimate_noise_floor(x, fs, len(x))
        rt60_best, rt60_candidates = dict(rt60=float("nan"), r2=float("nan"), range=None), []
        peak_norm = float(np.abs(x).max())
    else:
        noise_rms, slope, quiet, f, Pxx = estimate_noise_floor(x, fs, onset_idx)
        tail = x[onset_idx:]
        rt60_best, rt60_candidates = estimate_rt60_robust(tail, fs)
        peak_norm = float(np.abs(x).max())

    mic_lo, mic_hi = estimate_mic_passband(f, Pxx)

    return dict(
        wav_path=wav_path, sample_rate_hz=fs, duration_s=len(x)/fs,
        onset_time_s=(onset_idx/fs if onset_idx is not None else None),
        noise_rms=noise_rms, noise_psd_slope=slope,
        rt60_s=rt60_best["rt60"], rt60_r2=rt60_best["r2"], rt60_fit_range_db=rt60_best["range"],
        rt60_candidates=rt60_candidates,
        peak_norm=peak_norm, mic_lo_hz=mic_lo, mic_hi_hz=mic_hi,
    )


def print_report(est):
    print(f"\n{'Parameter':<28} {'Estimated value':>18}   Maps to gen_config.ini")
    print("-" * 72)
    print(f"{'Noise floor RMS':<28} {est['noise_rms']:>18.6g}   [calibration] real_noise_rms")
    print(f"{'Noise PSD slope (exponent)':<28} {est['noise_psd_slope']:>18.3f}   "
          f"[calibration] real_noise_slope")
    print(f"{'RT60, Schroeder (s)':<28} {est['rt60_s']:>18.3f}   [calibration] real_rt60")
    print(f"{'Peak amplitude (normalized)':<28} {est['peak_norm']:>18.6f}   "
          f"[calibration] real_peak_norm")
    print(f"{'Mic passband low (Hz)':<28} {est['mic_lo_hz']:>18.1f}   [adc] mic_lo  (approximate, see caveat)")
    print(f"{'Mic passband high (Hz)':<28} {est['mic_hi_hz']:>18.1f}   [adc] mic_hi  (approximate, see caveat)")

    if est["rt60_candidates"]:
        print(f"\nRT60 fit-quality check (a real room's decay isn't always a clean single")
        print(f"exponential -- low R^2 below means DON'T trust that window's RT60):")
        print(f"  {'Fit window (dB)':<18} {'RT60 (s)':>10} {'R^2':>8}")
        for r in est["rt60_candidates"]:
            marker = "  <- used" if r["range"] == est["rt60_fit_range_db"] else ""
            r2_str = f"{r['r2']:.3f}" if np.isfinite(r["r2"]) else "N/A"
            rt60_str = f"{r['rt60']:.3f}" if np.isfinite(r["rt60"]) else "N/A"
            print(f"  {str(r['range']):<18} {rt60_str:>10} {r2_str:>8}{marker}")
        if est["rt60_r2"] < 0.9:
            print(f"  WARNING: best R^2 = {est['rt60_r2']:.3f} -- this recording's decay "
                  f"isn't well described by a single RT60 number. Treat the estimate as "
                  f"approximate, and consider inspecting the Schroeder curve directly.")


def write_config_snippet(est, out_path):
    snippet = f"""\
# Estimated from: {est['wav_path']}
# (paste into gen_config.ini, replacing the existing [calibration] section)

[calibration]
real_noise_rms = {est['noise_rms']:.6g}
real_rt60 = {est['rt60_s']:.4f}
real_peak_norm = {est['peak_norm']:.6f}
real_noise_slope = {est['noise_psd_slope']:.3f}

[noise]
model = realistic
rt60 = {est['rt60_s']:.4f}
noise_rms_pa = {est['noise_rms']:.6g}
noise_slope = {est['noise_psd_slope']:.3f}

# Mic passband estimate is APPROXIMATE -- it mixes true mic/ADC response
# with environmental noise coloring. Verify against your actual hardware
# spec before trusting it; treat as a starting point, not ground truth.
[adc]
mic_lo = {est['mic_lo_hz']:.1f}
mic_hi = {est['mic_hi_hz']:.1f}
"""
    with open(out_path, "w") as f:
        f.write(snippet)
    print(f"\nWrote config snippet: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate noise/reverb/calibration parameters from a real recording.")
    parser.add_argument("wav_path", nargs="?", default=None,
                         help="Path to the real recording (.wav) -- single-channel mode")
    parser.add_argument("--event-threshold", type=float, default=8.0,
                         help="Onset detection threshold, x noise RMS (default: 8.0)")
    parser.add_argument("--config-out", default=None,
                         help="Write a ready-to-paste gen_config.ini snippet to this path")
    parser.add_argument("--check-channels", nargs="+", metavar="WAV",
                         help="Cross-check multiple per-mic .wav files of the SAME event "
                              "against their consensus mean, flag problem channels "
                              "(inverted polarity / low correlation), and recommend one. "
                              "E.g.: --check-channels rec_ch0.wav rec_ch1.wav ... rec_ch6.wav")
    args = parser.parse_args()

    if args.check_channels:
        check_channel_quality(args.check_channels)
        return

    if args.wav_path is None:
        parser.error("wav_path is required unless --check-channels is used")

    est = estimate_all(args.wav_path, event_thresh_mult=args.event_threshold)
    print_report(est)

    if args.config_out:
        write_config_snippet(est, args.config_out)


if __name__ == "__main__":
    main()
