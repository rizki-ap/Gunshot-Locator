#!/usr/bin/env python3
"""
signal_similarity.py
=====================
Standalone toolkit for quantifying how similar two acoustic signals are --
e.g. a simulated gunshot channel vs. a real recording, or any two .wav files.

Works as:
  1. A library:  `from signal_similarity import compare_signals`
  2. A CLI tool: `python signal_similarity.py sig1.wav sig2.wav`

No dependency on any specific notebook/pipeline -- only numpy/scipy.

Metrics reported
-----------------
- Pearson correlation coefficient (zero lag) -- THE standard academic
  parameter for signal similarity, bounded [-1, +1]. Cited across signal
  processing, seismology, and time-series literature generally (e.g.
  Bendat & Piersol, "Random Data: Analysis and Measurement Procedures").
  rho(x,y) = (1/(N-1)) * sum_i [(x_i-mean(x))/std(x)] * [(y_i-mean(y))/std(y)]
- Cross-correlation coefficient at its best-aligning lag -- the same
  parameter as a function of time shift tau, since two physically similar
  signals are rarely perfectly aligned to begin with; reduces to the above
  at tau=0.
- Magnitude-squared coherence (frequency-domain analogue, band-averaged) --
  the other measure academic sources consistently recommend alongside
  cross-correlation for signal similarity.
- Peak/noise SNR (amplitude-domain)
- PSD slope (dB/decade) + PSD shape correlation (spectral coloring, level-
  independent)
- RT60 via Schroeder (1965) backward energy integration
- Onset half-period (a cheap proxy for pulse duration, e.g. N-wave T_N/2)

A second suite (compare_signals_ml / --suite ml) adds the metrics standard
in audio-ML / TTS / vocoder evaluation literature (not classical acoustics --
see that section's docstring for the distinction): RMSE (time, aligned),
cross-correlation, Log-Spectral Distance, Mel-Spectrogram MSE/SSIM, MFCC
cosine similarity (naive AND event-windowed -- the naive version has a
documented failure mode on mostly-silent recordings), and DTW distance.

Usage
-----
  python signal_similarity.py real.wav sim.wav                  # both suites
  python signal_similarity.py real.wav sim.wav --suite ml        # just the ML suite
  python signal_similarity.py real.wav sim.wav --plot out_dir/

As a library:
  import numpy as np
  from signal_similarity import compare_signals, compare_signals_ml
  report = compare_signals(x1, fs1, x2, fs2, label1="Real", label2="Sim")
  report_ml = compare_signals_ml(x1, fs1, x2, fs2, label1="Real", label2="Sim")
"""

import argparse
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, coherence, resample_poly, correlate, stft as scipy_stft
from scipy.ndimage import uniform_filter
from scipy.fftpack import dct


# ---------------------------------------------------------------------------
# Loading
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


def match_sample_rates(x1, fs1, x2, fs2, target_fs=None):
    """Resample both signals to a common rate (needed for cross-correlation
    and coherence, which assume a shared sample rate). PSD/RT60/onset metrics
    don't need this since they're computed independently per signal."""
    if target_fs is None:
        target_fs = min(fs1, fs2)

    def _resample(x, fs):
        if fs == target_fs:
            return x
        from math import gcd
        g = gcd(int(fs), int(target_fs))
        return resample_poly(x, int(target_fs) // g, int(fs) // g)

    return _resample(x1, fs1), _resample(x2, fs2), target_fs


# ---------------------------------------------------------------------------
# Amplitude / SNR
# ---------------------------------------------------------------------------
def amplitude_stats(x, fs, noise_window_s=0.005):
    n = max(1, int(noise_window_s * fs))
    noise_rms = np.sqrt(np.mean(x[:n] ** 2))
    peak = np.abs(x).max()
    return dict(
        noise_rms=noise_rms,
        peak=peak,
        snr_db=20 * np.log10(peak / (noise_rms + 1e-12)),
    )


# ---------------------------------------------------------------------------
# Time-domain similarity: the (Pearson) cross-correlation coefficient
# ---------------------------------------------------------------------------
#
# THE standard academic parameter for signal similarity. Bounded in [-1, +1]:
# +1 = perfect linear match, -1 = perfect inverted match, 0 = uncorrelated.
# It's the cross-correlation function normalized so it reduces to the
# ordinary Pearson correlation coefficient at zero lag:
#
#     rho(x,y) = (1/(N-1)) * sum_i [(x_i - mean(x))/std(x)] * [(y_i - mean(y))/std(y)]
#
# References: Bendat & Piersol, "Random Data: Analysis and Measurement
# Procedures" (the standard signal-processing reference for this); widely
# used the same way in seismology (cross-correlation of waveforms),
# neuroscience/fMRI connectivity, and time-series similarity generally.
# Reported here at its best-aligning lag, since two physically-similar
# signals are rarely perfectly time-aligned to begin with.

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


def psd_shape_correlation(f1, Pxx1, f2, Pxx2, f_lo=100, f_hi=10000, n_bins=50):
    """Correlate log-PSD SHAPE (mean-removed) on a shared log-frequency grid --
    isolates spectral coloring from absolute level."""
    f_hi = min(f_hi, f1.max(), f2.max())
    common_f = np.logspace(np.log10(f_lo), np.log10(f_hi), n_bins)
    logP1 = np.interp(common_f, f1, 10 * np.log10(Pxx1 + 1e-20))
    logP2 = np.interp(common_f, f2, 10 * np.log10(Pxx2 + 1e-20))
    logP1 = logP1 - logP1.mean()
    logP2 = logP2 - logP2.mean()
    return float(np.corrcoef(logP1, logP2)[0, 1])


def band_averaged_coherence(x1, x2, fs, f_lo=100, f_hi=10000, nperseg=2048):
    """Magnitude-squared coherence, averaged over [f_lo, f_hi]. Requires both
    signals at the SAME sample rate and comparable length/alignment -- this
    is the strictest metric here since it's sensitive to phase, not just
    magnitude shape."""
    n = min(len(x1), len(x2))
    f, Cxy = coherence(x1[:n], x2[:n], fs=fs, nperseg=min(nperseg, n))
    mask = (f >= f_lo) & (f <= f_hi)
    return float(np.mean(Cxy[mask])) if mask.any() else np.nan


# ---------------------------------------------------------------------------
# Reverberation: Schroeder RT60
# ---------------------------------------------------------------------------
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


def event_shape_params(x, fs, onset_idx, win_s=0.010):
    """Half-period of the first zero-crossing after onset. A cheap duration
    proxy (e.g. comparable to N-wave T_N/2) -- not a substitute for fitting
    the actual expected pulse shape, but useful as a fast sanity check."""
    if onset_idx is None:
        return dict(peak=np.nan, half_period_ms=np.nan)
    seg = x[onset_idx: onset_idx + int(win_s * fs)]
    zc = np.where(np.diff(np.sign(seg)))[0]
    return dict(
        peak=float(seg[np.argmax(np.abs(seg))]) if len(seg) else np.nan,
        half_period_ms=(zc[0] / fs * 1e3) if len(zc) else np.nan,
    )


# ===========================================================================
# Audio-ML / TTS-style similarity suite
# ===========================================================================
# The metrics below (RMSE-time, LSD, Mel-Spec MSE/SSIM, MFCC cosine sim,
# DTW) are the standard evaluation battery used in speech-synthesis/vocoder
# literature (Tacotron/WaveNet-era TTS papers, singing-voice-synthesis,
# text-to-audio generation), not classical acoustics -- see the discussion
# in this module's usage notes. Included here because they're commonly
# requested alongside the classical metrics above, but note they were
# designed for continuous speech, not brief impulsive transients, and at
# least one of them (naive MFCC cosine similarity) has a specific failure
# mode on mostly-silent recordings -- see mfcc_cosine_similarity_naive().

def align_signals(x1, x2, fs, max_lag_s=0.05):
    """Shift x2 to best-align with x1 (via cross-correlation), then crop to
    the overlapping region. Returns (x1_aligned, x2_aligned, lag_s), where
    lag_s > 0 means x2's content arrives AFTER x1's."""
    a = x1 - x1.mean()
    b = x2 - x2.mean()
    norm = np.std(a) * np.std(b) * min(len(a), len(b))
    corr = correlate(a, b, mode="full") / (norm + 1e-20)
    raw_lags = np.arange(-len(b) + 1, len(a))
    max_lag_n = int(max_lag_s * fs)
    mask = np.abs(raw_lags) <= max_lag_n
    corr_m, raw_lags_m = corr[mask], raw_lags[mask]
    raw_lag = int(raw_lags_m[np.argmax(np.abs(corr_m))])
    lag = -raw_lag  # match cross_correlation_coefficient's convention

    if lag >= 0:
        a_al = x1[: len(x1) - lag] if lag > 0 else x1
        b_al = x2[lag:]
    else:
        a_al = x1[-lag:]
        b_al = x2[: len(x2) + lag] if lag < 0 else x2
    n = min(len(a_al), len(b_al))
    return a_al[:n], b_al[:n], lag / fs


def rmse_time(x1, x2):
    """Root-mean-square error, time domain. Extremely sensitive to
    misalignment -- a single-sample shift on an impulsive signal can
    dominate this. Always align signals first (see align_signals)."""
    n = min(len(x1), len(x2))
    return float(np.sqrt(np.mean((x1[:n] - x2[:n]) ** 2)))


def _stft_power(x, fs, n_fft=1024, hop=256):
    f, t, Z = scipy_stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    return f, t, np.abs(Z) ** 2


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_fft, fs, n_mels=40, fmin=0.0, fmax=None):
    """Standard HTK-style triangular mel filterbank (numpy/scipy only, no
    librosa dependency)."""
    if fmax is None:
        fmax = fs / 2
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    n_freq = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels + 1):
        f_lo, f_mid, f_hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f_lo, min(f_mid, n_freq)):
            if f_mid > f_lo:
                fbank[m - 1, k] = (k - f_lo) / (f_mid - f_lo)
        for k in range(f_mid, min(f_hi, n_freq)):
            if f_hi > f_mid:
                fbank[m - 1, k] = (f_hi - k) / (f_hi - f_mid)
    return fbank


def log_mel_spectrogram(x, fs, n_fft=1024, hop=256, n_mels=40, fmin=0.0, fmax=None, eps=1e-10):
    f, t, P = _stft_power(x, fs, n_fft, hop)
    fbank = mel_filterbank(n_fft, fs, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return np.log10(fbank @ P + eps), t


def log_spectral_distance(x1, x2, fs, n_fft=1024, hop=256, eps=1e-10):
    """Log-Spectral Distance (LSD), in dB. Standard defn: frame-wise RMS of
    the log-power spectral difference, averaged over frames. Lower = more
    similar. Signals should be time-aligned and at the same sample rate
    first (see align_signals / match_sample_rates)."""
    _, _, P1 = _stft_power(x1, fs, n_fft, hop)
    _, _, P2 = _stft_power(x2, fs, n_fft, hop)
    n_frames = min(P1.shape[1], P2.shape[1])
    log_diff = 10*np.log10(P1[:, :n_frames]+eps) - 10*np.log10(P2[:, :n_frames]+eps)
    return float(np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0))))


def mel_spectrogram_mse(x1, x2, fs, **kwargs):
    """MSE between log-mel spectrograms."""
    M1, _ = log_mel_spectrogram(x1, fs, **kwargs)
    M2, _ = log_mel_spectrogram(x2, fs, **kwargs)
    n = min(M1.shape[1], M2.shape[1])
    return float(np.mean((M1[:, :n] - M2[:, :n]) ** 2))


def ssim_2d(A, B, win_size=7):
    """Structural Similarity Index (Wang, Bovik, Sheikh & Simoncelli, 2004),
    with uniform (box) windows rather than the original Gaussian window --
    a common simplification, also used by e.g. scikit-image's default."""
    dyn_range = max(A.max(), B.max()) - min(A.min(), B.min())
    C1, C2 = (0.01*dyn_range)**2, (0.03*dyn_range)**2
    mu_A, mu_B = uniform_filter(A, size=win_size), uniform_filter(B, size=win_size)
    mu_A2, mu_B2, mu_AB = mu_A**2, mu_B**2, mu_A*mu_B
    sigma_A2 = uniform_filter(A*A, size=win_size) - mu_A2
    sigma_B2 = uniform_filter(B*B, size=win_size) - mu_B2
    sigma_AB = uniform_filter(A*B, size=win_size) - mu_AB
    num = (2*mu_AB + C1) * (2*sigma_AB + C2)
    den = (mu_A2 + mu_B2 + C1) * (sigma_A2 + sigma_B2 + C2)
    return float(np.mean(num / den))


def mel_spectrogram_ssim(x1, x2, fs, **kwargs):
    M1, _ = log_mel_spectrogram(x1, fs, **kwargs)
    M2, _ = log_mel_spectrogram(x2, fs, **kwargs)
    n = min(M1.shape[1], M2.shape[1])
    return ssim_2d(M1[:, :n], M2[:, :n])


def mfcc(x, fs, n_fft=1024, hop=256, n_mels=40, n_mfcc=13, **kwargs):
    """Mel-Frequency Cepstral Coefficients (Davis & Mermelstein, 1980):
    DCT-II of the log-mel spectrogram. Returns (coeffs, frame_times, hop_s)."""
    logmel, t = log_mel_spectrogram(x, fs, n_fft=n_fft, hop=hop, n_mels=n_mels, **kwargs)
    coeffs = dct(logmel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return coeffs, t, hop / fs


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def mfcc_cosine_similarity_naive(x1, fs1, x2, fs2, **mfcc_kwargs):
    """Whole-clip time-averaged MFCC vector, then cosine similarity.

    WARNING: for mostly-silent recordings with one brief event (like a
    gunshot), this is dominated by the many near-identical, near-zero
    silent frames, and can read deceptively close to 1.0 even when the
    actual event sounds nothing alike -- verified directly: two signals
    with a 2kHz vs. an 8kHz burst embedded in equal silence score 0.9997
    here despite being clearly different events. Kept in this module
    specifically to make that failure mode visible/reproducible; use
    mfcc_cosine_similarity_event() for a metric that actually reflects
    event similarity."""
    C1, _, _ = mfcc(x1, fs1, **mfcc_kwargs)
    C2, _, _ = mfcc(x2, fs2, **mfcc_kwargs)
    return cosine_similarity(C1.mean(axis=1), C2.mean(axis=1))


def mfcc_cosine_similarity_event(x1, fs1, x2, fs2, onset1_s, onset2_s,
                                  window_s=0.020, **mfcc_kwargs):
    """MFCC cosine similarity restricted to a window around each signal's
    own event onset -- excludes the silent majority of the clip, so it
    actually measures event-timbre similarity rather than being dominated
    by shared silence. onset1_s/onset2_s: onset time (s) in each signal,
    e.g. from find_loudest_onset()."""
    C1, t1, _ = mfcc(x1, fs1, **mfcc_kwargs)
    C2, t2, _ = mfcc(x2, fs2, **mfcc_kwargs)
    m1 = (t1 >= onset1_s - window_s/4) & (t1 <= onset1_s + window_s)
    m2 = (t2 >= onset2_s - window_s/4) & (t2 <= onset2_s + window_s)
    if not m1.any() or not m2.any():
        return np.nan
    return cosine_similarity(C1[:, m1].mean(axis=1), C2[:, m2].mean(axis=1))


def dtw_distance(seq1, seq2):
    """Dynamic Time Warping (Sakoe & Chiba, 1978) with Euclidean local cost,
    on feature-FRAME sequences (e.g. MFCC) -- not raw audio samples, which
    would be computationally infeasible and isn't how DTW is used in the
    audio-ML literature. Normalized by warping-path length so it's
    comparable across signals of different duration. seq1/seq2 shape:
    (n_features, n_frames)."""
    A, B = seq1.T, seq2.T
    n, m = len(A), len(B)
    dist = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        row_prev, row = D[i - 1], D[i]
        for j in range(1, m + 1):
            row[j] = dist[i-1, j-1] + min(row_prev[j], row[j-1], row_prev[j-1])
    i, j, path_len = n, m, 0
    while i > 0 and j > 0:
        path_len += 1
        choices = (D[i-1, j-1], D[i-1, j], D[i, j-1])
        move = int(np.argmin(choices))
        i, j = (i-1, j-1) if move == 0 else (i-1, j) if move == 1 else (i, j-1)
    path_len += i + j
    return float(D[n, m] / max(path_len, 1))


def compare_signals_ml(x1, fs1, x2, fs2, label1="Signal 1", label2="Signal 2",
                        max_lag_s=0.05, mfcc_window_s=0.020, verbose=True):
    """Run the audio-ML/TTS-style similarity suite: RMSE(time), cross-
    correlation, LSD, Mel-Spec MSE, Mel-Spec SSIM, MFCC cosine similarity
    (both naive and event-windowed), DTW distance."""
    x1r, x2r, fs_c = match_sample_rates(x1, fs1, x2, fs2)
    a_al, b_al, lag_s = align_signals(x1r, x2r, fs_c, max_lag_s=max_lag_s)

    report = {}
    report["rmse_time_aligned"] = rmse_time(a_al, b_al)
    report["cross_correlation"] = pearson_correlation_coefficient(a_al, b_al)
    report["lsd_db"] = log_spectral_distance(a_al, b_al, fs_c)
    report["mel_spec_mse"] = mel_spectrogram_mse(a_al, b_al, fs_c)
    report["mel_spec_ssim"] = mel_spectrogram_ssim(a_al, b_al, fs_c)

    onset1_idx, _ = find_loudest_onset(x1, fs1)
    onset2_idx, _ = find_loudest_onset(x2, fs2)
    onset1_s = onset1_idx / fs1 if onset1_idx is not None else 0.0
    onset2_s = onset2_idx / fs2 if onset2_idx is not None else 0.0
    report["mfcc_cosine_naive"] = mfcc_cosine_similarity_naive(x1, fs1, x2, fs2)
    report["mfcc_cosine_event"] = mfcc_cosine_similarity_event(
        x1, fs1, x2, fs2, onset1_s, onset2_s, window_s=mfcc_window_s)

    C1, _, _ = mfcc(a_al, fs_c)
    C2, _, _ = mfcc(b_al, fs_c)
    report["dtw_distance"] = dtw_distance(C1, C2)

    if verbose:
        w = 26
        print(f"\n{'--- Audio-ML / TTS-style suite ---':<{w}} {label1:>16} {label2:>16}")
        print(f"(aligned at lag {lag_s*1e3:+.3f} ms before computing)")
        print(f"{'RMSE (time, aligned)':<{w}} {'':>16} {report['rmse_time_aligned']:>16.4f}  (lower=better)")
        print(f"{'Cross-Correlation':<{w}} {'':>16} {report['cross_correlation']:>16.4f}  (closer to 1=better)")
        print(f"{'Log-Spectral Dist (dB)':<{w}} {'':>16} {report['lsd_db']:>16.4f}  (lower=better)")
        print(f"{'Mel-Spec MSE':<{w}} {'':>16} {report['mel_spec_mse']:>16.4f}  (lower=better)")
        print(f"{'Mel-Spec SSIM':<{w}} {'':>16} {report['mel_spec_ssim']:>16.4f}  (closer to 1=better)")
        print(f"{'MFCC Cosine Sim (naive)':<{w}} {'':>16} {report['mfcc_cosine_naive']:>16.4f}  "
              f"(WARNING: see docstring -- often misleadingly high)")
        print(f"{'MFCC Cosine Sim (event)':<{w}} {'':>16} {report['mfcc_cosine_event']:>16.4f}  "
              f"(closer to 1=better, more trustworthy)")
        print(f"{'DTW Distance':<{w}} {'':>16} {report['dtw_distance']:>16.4f}  (lower=better)")

    return report


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------
def compare_signals(x1, fs1, x2, fs2, label1="Signal 1", label2="Signal 2",
                     max_lag_s=0.05, verbose=True):
    """Run the full comparison and return a dict of metrics. Prints a table
    if verbose=True."""
    report = {}

    s1, s2 = amplitude_stats(x1, fs1), amplitude_stats(x2, fs2)
    report["snr_db_1"], report["snr_db_2"] = s1["snr_db"], s2["snr_db"]

    f1, P1 = psd_welch(x1, fs1)
    f2, P2 = psd_welch(x2, fs2)
    report["psd_slope_1"] = psd_slope_fit(f1, P1)
    report["psd_slope_2"] = psd_slope_fit(f2, P2)
    report["psd_shape_corr"] = psd_shape_correlation(f1, P1, f2, P2)

    report["rt60_1"] = estimate_rt60_schroeder(x1, fs1)
    report["rt60_2"] = estimate_rt60_schroeder(x2, fs2)

    onset1, _ = find_loudest_onset(x1, fs1)
    onset2, _ = find_loudest_onset(x2, fs2)
    shp1 = event_shape_params(x1, fs1, onset1)
    shp2 = event_shape_params(x2, fs2, onset2)
    report["onset_half_period_ms_1"] = shp1["half_period_ms"]
    report["onset_half_period_ms_2"] = shp2["half_period_ms"]

    # Metrics requiring a shared sample rate
    x1r, x2r, fs_common = match_sample_rates(x1, fs1, x2, fs2)

    n_common = min(len(x1r), len(x2r))
    report["pearson_r_zero_lag"] = pearson_correlation_coefficient(x1r[:n_common], x2r[:n_common])

    xc, lag_s = cross_correlation_coefficient(x1r, x2r, fs_common, max_lag_s=max_lag_s)
    report["cross_corr_coeff_best_lag"] = xc
    report["best_lag_s"] = lag_s
    report["coherence_band_avg"] = band_averaged_coherence(x1r, x2r, fs_common)

    if verbose:
        w = 26
        print(f"{'Metric':<{w}} {label1:>16} {label2:>16}")
        print("-" * (w + 34))
        print(f"{'Peak/noise SNR (dB)':<{w}} {s1['snr_db']:>16.2f} {s2['snr_db']:>16.2f}")
        print(f"{'PSD slope (dB/decade)':<{w}} {report['psd_slope_1']:>16.2f} {report['psd_slope_2']:>16.2f}")
        print(f"{'RT60, Schroeder (s)':<{w}} {report['rt60_1']:>16.3f} {report['rt60_2']:>16.3f}")
        print(f"{'Onset half-period (ms)':<{w}} {shp1['half_period_ms']:>16.3f} {shp2['half_period_ms']:>16.3f}")
        print("-" * (w + 34))
        print(f"{'*Pearson r (zero lag)*':<{w}} {report['pearson_r_zero_lag']:>16.3f}  "
              f"<- THE standard academic similarity parameter")
        print(f"{'Cross-corr. coeff (best lag)':<{w}} {xc:>16.3f}  (at lag {lag_s*1e3:+.3f} ms)")
        print(f"{'PSD shape correlation':<{w}} {report['psd_shape_corr']:>16.3f}  (1.0 = identical spectral shape)")
        print(f"{'Band-avg. coherence':<{w}} {report['coherence_band_avg']:>16.3f}  (1.0 = perfectly coherent)")

    return report


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------
def plot_comparison(x1, fs1, x2, fs2, label1, label2, out_dir):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    t1 = np.arange(len(x1)) / fs1
    t2 = np.arange(len(x2)) / fs2
    axes[0].plot(t1, x1, lw=0.5, label=label1, alpha=0.8)
    axes[0].plot(t2, x2, lw=0.5, label=label2, alpha=0.8)
    axes[0].set_title("Waveforms (normalized, not time-aligned)")
    axes[0].set_xlabel("Time (s)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    f1, P1 = psd_welch(x1, fs1)
    f2, P2 = psd_welch(x2, fs2)
    axes[1].semilogx(f1, 10*np.log10(P1+1e-20), label=label1)
    axes[1].semilogx(f2, 10*np.log10(P2+1e-20), label=label2)
    axes[1].set_title("Power Spectral Density")
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("dB")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    t1e, edc1 = schroeder_edc_db(x1, fs1)
    t2e, edc2 = schroeder_edc_db(x2, fs2)
    axes[2].plot(t1e, edc1, label=label1)
    axes[2].plot(t2e, edc2, label=label2)
    axes[2].set_title("Schroeder Energy Decay Curve")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("dB")
    axes[2].set_ylim([-80, 5]); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "signal_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare similarity between two acoustic signals (.wav files)."
    )
    parser.add_argument("file1", help="Path to first .wav file")
    parser.add_argument("file2", help="Path to second .wav file")
    parser.add_argument("--label1", default=None, help="Label for file1 in output")
    parser.add_argument("--label2", default=None, help="Label for file2 in output")
    parser.add_argument("--max-lag-ms", type=float, default=50.0,
                         help="Max lag (ms) to search for best-lag correlation (default: 50)")
    parser.add_argument("--plot", metavar="OUT_DIR", default=None,
                         help="If set, save comparison plots to this directory")
    parser.add_argument("--suite", choices=["classic", "ml", "both"], default="both",
                         help="Which metric suite to run (default: both)")
    args = parser.parse_args()

    label1 = args.label1 or args.file1
    label2 = args.label2 or args.file2

    fs1, x1 = load_wav_normalized(args.file1)
    fs2, x2 = load_wav_normalized(args.file2)

    print(f"{label1}: fs={fs1} Hz, duration={len(x1)/fs1:.3f} s")
    print(f"{label2}: fs={fs2} Hz, duration={len(x2)/fs2:.3f} s\n")

    if args.suite in ("classic", "both"):
        compare_signals(x1, fs1, x2, fs2, label1=label1, label2=label2,
                         max_lag_s=args.max_lag_ms / 1000.0)
    if args.suite in ("ml", "both"):
        compare_signals_ml(x1, fs1, x2, fs2, label1=label1, label2=label2,
                            max_lag_s=args.max_lag_ms / 1000.0)

    if args.plot:
        plot_comparison(x1, fs1, x2, fs2, label1, label2, args.plot)


if __name__ == "__main__":
    main()