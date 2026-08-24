#!/usr/bin/env python3
"""
plot_wav_channels.py
=======================
Plots every channel of a multichannel .wav file -- one row per channel,
full-duration view alongside an auto-zoomed view around the loudest event
(a plain full-duration plot of a mostly-silent gunshot-length recording is
nearly useless -- the event is a few ms out of hundreds, so it looks like a
flat line with a hairline spike unless you also zoom in).

Handles any channel count (1..N) and any common WAV dtype:
  - Integer PCM (int16/int32/uint8): normalized to the standard +-1 audio
    convention using the dtype's full-scale range.
  - Float32/float64: plotted in RAW units, NOT renormalized -- this project
    also produces float32 WAVs holding actual physical units (e.g. Pa from
    generate_clean_signal.py), and silently rescaling those would destroy
    that information. Use --normalize to force peak-normalization anyway.

Usage
-----
  python plot_wav_channels.py signal.wav
  python plot_wav_channels.py signal.wav -o out.png
  python plot_wav_channels.py signal.wav --start 0.24 --end 0.30   # explicit zoom window
  python plot_wav_channels.py signal.wav --zoom-window 0.010       # +-10ms auto-zoom (default 20ms)
  python plot_wav_channels.py signal.wav --overlay                 # all channels on one axes
  python plot_wav_channels.py signal.wav --normalize                # force peak-normalize floats too
  python plot_wav_channels.py signal.wav --no-zoom                  # full-duration view only
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile


COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4',
          '#795548', '#607D8B']

INT_FULL_SCALE = {
    np.dtype('int16'): 2**15,
    np.dtype('int32'): 2**31,
    np.dtype('uint8'): 2**7,
}


def load_wav(path, force_normalize=False):
    fs, data = wavfile.read(path)
    if data.ndim == 1:
        data = data[:, None]
    n_channels = data.shape[1]

    dtype = data.dtype
    is_float = np.issubdtype(dtype, np.floating)

    if is_float and not force_normalize:
        signals = data.astype(np.float64)
        unit_label = "Amplitude (raw float units)"
    elif is_float and force_normalize:
        peak = np.abs(data).max()
        signals = data.astype(np.float64) / (peak + 1e-20)
        unit_label = "Amplitude (peak-normalized)"
    else:
        full_scale = INT_FULL_SCALE.get(dtype, 2**(data.itemsize*8 - 1))
        signals = data.astype(np.float64) / full_scale
        unit_label = "Amplitude (normalized, +-1 = full scale)"

    return signals, fs, n_channels, dtype, unit_label


def channel_stats(sig, fs, noise_window_s=0.005):
    n = max(1, int(noise_window_s * fs))
    noise_rms = np.sqrt(np.mean(sig[:n] ** 2))
    peak = np.abs(sig).max()
    peak_idx = int(np.argmax(np.abs(sig)))
    return dict(peak=peak, peak_idx=peak_idx, peak_time_s=peak_idx / fs,
                noise_rms=noise_rms,
                snr_db=20*np.log10(peak/(noise_rms+1e-12)) if noise_rms > 0 else np.nan)


def plot_channels(signals, fs, n_channels, unit_label, title, out_path,
                   start_s=None, end_s=None, zoom_window_s=0.020,
                   overlay=False, no_zoom=False):
    t = np.arange(signals.shape[0]) / fs
    duration = t[-1] if len(t) else 0.0

    # Determine the zoom window: explicit --start/--end wins; otherwise
    # auto-center on the loudest sample across ALL channels (shared window,
    # so relative arrival-time offsets between channels stay visible).
    stats = [channel_stats(signals[:, i], fs) for i in range(n_channels)]
    if start_s is not None and end_s is not None:
        zoom_lo, zoom_hi = start_s, end_s
    else:
        loudest_ch = int(np.argmax([s["peak"] for s in stats]))
        center = stats[loudest_ch]["peak_time_s"]
        zoom_lo, zoom_hi = max(0.0, center - zoom_window_s), min(duration, center + zoom_window_s)

    n_cols = 1 if (no_zoom or overlay) else 2
    n_rows = 1 if overlay else n_channels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, max(2.2*n_rows, 3)),
                              squeeze=False)

    def _plot_one(ax, idx_list, mask):
        for i in idx_list:
            col = COLORS[i % len(COLORS)]
            ax.plot(t[mask], signals[mask, i], lw=0.6, color=col,
                    label=f"Ch {i}" if overlay else None)

    if overlay:
        _plot_one(axes[0, 0], range(n_channels), slice(None))
        axes[0, 0].set_title(f"All channels -- full duration", fontsize=10, fontweight='bold')
        axes[0, 0].legend(fontsize=8, ncol=n_channels)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel(unit_label)
        axes[0, 0].grid(alpha=0.3)
        if not no_zoom:
            zmask = (t >= zoom_lo) & (t <= zoom_hi)
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            for i in range(n_channels):
                ax2.plot(t[zmask], signals[zmask, i], lw=0.8, color=COLORS[i % len(COLORS)],
                         label=f"Ch {i}")
            ax2.set_title(f"All channels -- zoom [{zoom_lo:.4f}, {zoom_hi:.4f}] s",
                          fontsize=10, fontweight='bold')
            ax2.legend(fontsize=8, ncol=n_channels)
            ax2.set_xlabel("Time (s)"); ax2.set_ylabel(unit_label)
            ax2.set_xlim(zoom_lo, zoom_hi)
            ax2.grid(alpha=0.3)
            fig2.suptitle(title, fontsize=11)
            fig2.tight_layout()
            zoom_path = out_path.rsplit(".", 1)[0] + "_zoom." + out_path.rsplit(".", 1)[1]
            fig2.savefig(zoom_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved: {zoom_path}")
    else:
        for i in range(n_channels):
            col = COLORS[i % len(COLORS)]
            s = stats[i]
            ax = axes[i, 0]
            ax.plot(t, signals[:, i], lw=0.5, color=col)
            ax.set_ylabel(f"Ch {i}", fontsize=9, fontweight='bold', color=col)
            ax.grid(alpha=0.25)
            ax.text(0.01, 0.92, f"peak={s['peak']:.4g}  " +
                    (f"SNR={s['snr_db']:.1f}dB" if np.isfinite(s['snr_db']) else "SNR=N/A (no noise floor)"),
                    transform=ax.transAxes, fontsize=7.5, va='top',
                    bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7))
            if i == 0:
                ax.set_title("Full duration", fontsize=10, fontweight='bold')
            if i == n_channels - 1:
                ax.set_xlabel("Time (s)")

            if n_cols == 2:
                axz = axes[i, 1]
                zmask = (t >= zoom_lo) & (t <= zoom_hi)
                axz.plot(t[zmask], signals[zmask, i], lw=0.8, color=col)
                if zoom_lo <= stats[i]["peak_time_s"] <= zoom_hi:
                    axz.axvline(stats[i]["peak_time_s"], color='gray', ls=':', lw=0.8, alpha=0.7)
                axz.set_xlim(zoom_lo, zoom_hi)   # explicit -- axvline/empty data must not stretch this
                axz.grid(alpha=0.25)
                if i == 0:
                    axz.set_title(f"Zoom [{zoom_lo:.4f}, {zoom_hi:.4f}] s", fontsize=10, fontweight='bold')
                if i == n_channels - 1:
                    axz.set_xlabel("Time (s)")

        axes[0, 0].set_ylabel(f"Ch 0\n{unit_label}", fontsize=8)

    fig.suptitle(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats, zoom_lo, zoom_hi


def main():
    parser = argparse.ArgumentParser(description="Plot every channel of a multichannel .wav file.")
    parser.add_argument("wav_path", help="Path to the .wav file")
    parser.add_argument("-o", "--output", default=None,
                         help="Output image path (default: <wav_basename>_channels.png)")
    parser.add_argument("--start", type=float, default=None, help="Zoom window start (s)")
    parser.add_argument("--end", type=float, default=None, help="Zoom window end (s)")
    parser.add_argument("--zoom-window", type=float, default=0.020,
                         help="+- half-width (s) of the auto-zoom window around the loudest "
                              "sample, used when --start/--end aren't given (default: 0.020)")
    parser.add_argument("--overlay", action="store_true",
                         help="Plot all channels on one shared axes instead of stacked rows")
    parser.add_argument("--normalize", action="store_true",
                         help="Force peak-normalization even for float-format WAVs")
    parser.add_argument("--no-zoom", action="store_true",
                         help="Skip the auto-zoomed view, full-duration plot only")
    args = parser.parse_args()

    signals, fs, n_channels, dtype, unit_label = load_wav(args.wav_path, args.normalize)
    duration = signals.shape[0] / fs

    print(f"{args.wav_path}: fs={fs} Hz, channels={n_channels}, dtype={dtype}, "
          f"duration={duration:.4f} s")

    out_path = args.output or (args.wav_path.rsplit(".wav", 1)[0] + "_channels.png")
    title = f"{args.wav_path}  |  {n_channels}ch @ {fs} Hz, {dtype}, {duration:.3f} s"

    stats, zoom_lo, zoom_hi = plot_channels(
        signals, fs, n_channels, unit_label, title, out_path,
        start_s=args.start, end_s=args.end, zoom_window_s=args.zoom_window,
        overlay=args.overlay, no_zoom=args.no_zoom)

    print(f"\n{'Ch':<4} {'Peak':>12} {'Peak time (s)':>14} {'Noise RMS':>12} {'SNR (dB)':>10}")
    for i, s in enumerate(stats):
        snr_str = f"{s['snr_db']:.2f}" if np.isfinite(s['snr_db']) else "N/A"
        print(f"{i:<4} {s['peak']:>12.5g} {s['peak_time_s']:>14.5f} "
              f"{s['noise_rms']:>12.5g} {snr_str:>10}")
    if not args.no_zoom:
        print(f"\nAuto-zoom window: [{zoom_lo:.4f}, {zoom_hi:.4f}] s "
              f"(centered on loudest sample across all channels)")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
