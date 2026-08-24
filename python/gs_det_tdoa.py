#!/usr/bin/env python3
"""
gs_det_tdoa.py
================
DETECTION STAGE 2. Reads Stage 1's prepared signal + onset detections and
computes GCC-PHAT time-difference-of-arrival for all 6 microphone pairs
(C(4,2)) -- separately for the shockwave and muzzle blast events.

Critical design point: windows are extracted around a SHARED reference
time (one reference channel's own detected onset), NOT each channel's own
independently-detected onset. Centering each channel's window on its own
onset silently cancels out the very TDOA this step exists to measure --
GCC-PHAT would then always report tau=0 regardless of true bearing. The
reference channel is chosen as whichever channel had the most reliable
detection in Stage 1.

Usage
-----
  python gs_det_tdoa.py det_result_prepared.json                # det_config.ini
  python gs_det_tdoa.py det_result_prepared.json my_det_config.ini
"""

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
from scipy.io import wavfile

from gs_det_signal_prepare import load_det_config


def gcc_phat(x1, x2, fs, interp_factor=16, max_lag_s=0.001):
    n = len(x1)
    X1 = np.fft.rfft(x1, n=n)
    X2 = np.fft.rfft(x2, n=n)
    G = X2 * np.conj(X1)
    G /= (np.abs(G) + 1e-12)
    nf = n * interp_factor
    cc = np.fft.fftshift(np.fft.irfft(G, n=nf))
    lags = np.arange(-nf // 2, nf // 2) / (fs * interp_factor)
    mask = np.abs(lags) <= max_lag_s
    idx = np.argmax(cc[mask])
    return float(lags[mask][idx])


def extract_window(signal, t_axis, t_center, t_pre, t_post):
    mask = (t_axis >= t_center - t_pre) & (t_axis <= t_center + t_post)
    return signal[mask]


def pick_reference_channel(per_channel):
    """Prefer a channel with both SW and reliable MB detected; fall back to
    SW-only if none have reliable MB."""
    candidates = [c for c in per_channel if c.get("sw_detected") and c.get("mb_detected") and c.get("mb_reliable")]
    if not candidates:
        candidates = [c for c in per_channel if c.get("sw_detected")]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["channel"])


def main():
    parser = argparse.ArgumentParser(
        description="Detection Stage 2: GCC-PHAT TDOA for all 6 mic pairs (SW and MB).")
    parser.add_argument("prepared_json", help="Path to Stage 1's *_prepared.json")
    parser.add_argument("config", nargs="?", default="det_config.ini",
                         help="Path to detection .ini config (default: det_config.ini)")
    parser.add_argument("-o", "--output", default=None,
                         help="Output basename (overrides output.basename in config)")
    args = parser.parse_args()

    cfg = load_det_config(args.config)
    if args.output:
        cfg["output"]["basename"] = args.output

    with open(args.prepared_json) as f:
        prep = json.load(f)

    fs, data = wavfile.read(prep["prepared_wav"])
    n_channels = data.shape[1]
    signals = [data[:, i].astype(np.float64) for i in range(n_channels)]
    t_axis = np.arange(len(signals[0])) / fs

    ref = pick_reference_channel(prep["per_channel"])
    if ref is None:
        raise RuntimeError("No channel had a usable SW detection -- cannot compute TDOA.")
    ref_ch = ref["channel"]
    print(f"[Det Stage 2] Reference channel: {ref_ch} "
          f"(t_sw={ref['t_sw']:.5f}s, mb_reliable={ref.get('mb_reliable')})")

    tw = cfg["tdoa"]
    pairs = list(combinations(range(n_channels), 2))

    sw_windows = [extract_window(signals[i], t_axis, ref["t_sw"],
                                  tw["sw_window_pre_s"], tw["sw_window_post_s"]) for i in range(n_channels)]
    tdoa_sw = {}
    for (a, b) in pairs:
        tau = gcc_phat(sw_windows[a], sw_windows[b], fs,
                        interp_factor=tw["interp_factor"], max_lag_s=tw["max_lag_s"])
        tdoa_sw[f"{a},{b}"] = tau

    tdoa_mb = {}
    mb_used = ref.get("mb_detected") and ref.get("mb_reliable")
    if mb_used:
        mb_windows = [extract_window(signals[i], t_axis, ref["t_mb"],
                                      tw["mb_window_pre_s"], tw["mb_window_post_s"]) for i in range(n_channels)]
        for (a, b) in pairs:
            tau = gcc_phat(mb_windows[a], mb_windows[b], fs,
                            interp_factor=tw["interp_factor"], max_lag_s=tw["max_lag_s"])
            tdoa_mb[f"{a},{b}"] = tau
    else:
        print("[Det Stage 2] WARNING: reference channel's MB detection wasn't reliable -- "
              "skipping MB TDOA (would be meaningless windowed around a bad onset guess).")
    print("SW TDOA (us):")
    for k, v in tdoa_sw.items():
        print(f"  ({k}): {v*1e6:+.3f}")
    if tdoa_mb:
        print("MB TDOA (us):")
        for k, v in tdoa_mb.items():
            print(f"  ({k}): {v*1e6:+.3f}")

    # Consistency check: for a truly consistent set, tau(1,2) should ~= tau(0,2)-tau(0,1), etc.
    # Reported for transparency, not used to "correct" anything.
    consistency_sw = {}
    for (a, b) in pairs:
        if a == 0:
            continue
        expected = tdoa_sw.get(f"0,{b}", 0) - tdoa_sw.get(f"0,{a}", 0)
        actual = tdoa_sw.get(f"{a},{b}", 0)
        consistency_sw[f"{a},{b}"] = float(actual - expected)

    out_json = f"{cfg['output']['basename']}_tdoa.json"
    dt_sw_to_mb = (ref["t_mb"] - ref["t_sw"]) if mb_used else None
    meta = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        stage="tdoa",
        source_prepared_json=args.prepared_json,
        reference_channel=ref_ch,
        sample_rate_hz=fs,
        n_channels=n_channels,
        mb_available=bool(tdoa_mb),
        dt_sw_to_mb_s=dt_sw_to_mb,
        dp_sw_ref_channel=ref.get("dP_sw"),
        t_n_sw_ref_channel=ref.get("T_N"),
        dp_mb_ref_channel=ref.get("dP_mb"),
        tdoa_sw_s=tdoa_sw,
        tdoa_mb_s=tdoa_mb,
        sw_consistency_residual_s=consistency_sw,
        config_used=cfg,
    )
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Det Stage 2] Wrote {out_json}")


if __name__ == "__main__":
    main()
