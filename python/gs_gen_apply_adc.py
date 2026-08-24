#!/usr/bin/env python3
"""
gs_gen_apply_adc.py
==============
STAGE 3 of 3. Reads the noisy per-mic signal written by gs_gen_add_noise.py and
applies everything that's a property of your SENSOR HARDWARE rather than
the environment: mic frequency response (passband), sensitivity calibration
against a real-recording target, and ADC bit-depth quantization. Writes the
final 4-channel .wav.

Why this is its own stage, not bundled into gs_gen_add_noise.py: mic/ADC response
doesn't change when the environment does, and vice versa. Keeping them
separate means you can run the SAME noisy environment through different
hardware configs, or the SAME hardware config across different
environments, without re-deriving anything -- exactly the same reasoning
Stage 1 vs Stage 2 already followed.

Usage
-----
  python gs_gen_apply_adc.py gunshot_sim_noisy.wav                     # gen_config.ini
  python gs_gen_apply_adc.py gunshot_sim_noisy.wav my_config.ini
  python gs_gen_apply_adc.py gunshot_sim_noisy.wav my_config.ini -o my_shot

Output
------
  <output>.wav   4-channel, 16-bit PCM -- the final "received" signal
  <output>.json  full chain of ground truth + noise + ADC processing details

Only reads the [adc], [calibration], and [output] sections of the config --
geometry/bullet/trajectory/noise don't matter here, they're already baked
into the noisy .wav from Stages 1-2.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.io import wavfile

import gs_gen_physic as gp


def load_noisy_wav(wav_path, json_path=None):
    """Load Stage 2's float32 noisy .wav (+ its .json for metadata passthrough,
    if present -- looked up automatically by convention if not given)."""
    fs, data = wavfile.read(wav_path)
    if data.dtype != np.float32:
        print(f"WARNING: expected float32 noisy .wav, got {data.dtype}. "
              f"Proceeding, but check this is really a Stage-2 output.")
    signals = [data[:, i].astype(np.float64) for i in range(data.shape[1])]

    if json_path is None:
        json_path = wav_path.rsplit(".wav", 1)[0] + ".json"
        if not json_path.endswith("_noisy.json") and wav_path.endswith("_noisy.wav"):
            json_path = wav_path[:-len("_noisy.wav")] + "_noisy.json"
    noisy_meta = None
    try:
        with open(json_path) as f:
            noisy_meta = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: no companion metadata file found at '{json_path}' -- "
              f"output .json will be missing upstream ground-truth/noise info.")

    return signals, fs, noisy_meta


def apply_adc(pa_signals, fs, cfg):
    """Mic response -> per-channel calibration -> shared-ADC effects ->
    quantization. mic_bandpass is applied UNIFORMLY (previously the 'simple'
    noise model accidentally skipped it -- fixed in the 3-stage split).

    Two calibration modes:
      - analog_chain.enabled = false (default): PEAK-MATCHING -- rescale so
        this signal's peak matches calibration.real_peak_norm. Always avoids
        clipping by construction. Uses ONE shared scale for all channels
        (peak-matching can't represent per-channel hardware differences).
      - analog_chain.enabled = true: GAIN-CHAIN -- compute the scale from
        mic sensitivity, preamp gain, and ADC Vref, PER CHANNEL (see
        cfg["analog_chain_per_channel"] -- falls back to [analog_chain]'s
        nominal values for any channel without an override). Absolute
        calibration, so it can genuinely clip if the hardware can't handle
        this signal's SPL.

    Shared-ADC effects (only meaningful in gain-chain mode, where per-
    channel absolute levels are physically real numbers, not just relative
    scalings): channel-to-channel gain/offset mismatch, and -- the big one
    for a TDOA system -- multiplexed-ADC timing skew, where channel i is
    delayed by i * conversion_time relative to channel 0."""
    adc, cal, achain = cfg["adc"], cfg["calibration"], cfg["analog_chain"]
    achain_pc, mchan = cfg["analog_chain_per_channel"], cfg["adc_multi_channel"]

    filtered = [gp.mic_bandpass(s, fs, adc["mic_lo"], adc["mic_hi"]) for s in pa_signals]
    n_ch = len(filtered)
    peak_pa = max(np.abs(s).max() for s in filtered)

    if achain["enabled"]:
        pa_to_norm_per_ch = [
            gp.gain_chain_scale(achain_pc[i]["mic_sensitivity_mv_per_pa"] / 1000.0,
                                 achain_pc[i]["preamp_gain_db"], achain["adc_vref_peak_v"])
            for i in range(n_ch)
        ]
        calibration_mode = "gain_chain"
    else:
        shared_scale = cal["real_peak_norm"] / peak_pa if peak_pa > 0 else 1.0
        pa_to_norm_per_ch = [shared_scale] * n_ch
        calibration_mode = "peak_match"

    normalized = [filtered[i] * pa_to_norm_per_ch[i] for i in range(n_ch)]

    # -- Shared-ADC effects (gain-chain mode only -- see docstring) ----------
    mismatch_applied = None
    if achain["enabled"]:
        rng = np.random.default_rng(mchan["mismatch_seed"])
        gain_mismatch = rng.uniform(-mchan["gain_mismatch_pct"], mchan["gain_mismatch_pct"], n_ch) / 100.0
        offset_mismatch_v = rng.uniform(-mchan["offset_mismatch_mv"], mchan["offset_mismatch_mv"], n_ch) / 1000.0
        offset_mismatch_norm = offset_mismatch_v / achain["adc_vref_peak_v"]

        for i in range(n_ch):
            normalized[i] = normalized[i] * (1.0 + gain_mismatch[i]) + offset_mismatch_norm[i]

        skew_applied_us = [0.0] * n_ch
        if mchan["type"] == "multiplexed":
            skew_samples_per_ch = mchan["conversion_time_us"] * 1e-6 * fs
            for i in range(n_ch):
                delay_samples = i * skew_samples_per_ch
                normalized[i] = gp.fractional_delay(normalized[i], delay_samples)
                skew_applied_us[i] = i * mchan["conversion_time_us"]

        mismatch_applied = dict(gain_mismatch_pct=gain_mismatch.tolist() if hasattr(gain_mismatch, "tolist") else list(gain_mismatch),
                                 offset_mismatch_mv=(offset_mismatch_v*1000).tolist(),
                                 adc_type=mchan["type"],
                                 timing_skew_us=skew_applied_us)
        gain_mismatch = list(gain_mismatch)  # ensure JSON-serializable below

    would_clip = any(np.abs(s).max() > 1.0 for s in normalized) if achain["enabled"] else False
    final_signals = [gp.quantize(s, bits=adc["bit_depth"]) for s in normalized]

    adc_meta = dict(calibration_mode=calibration_mode,
                     peak_pa_before_calibration=float(peak_pa),
                     pa_to_norm_scale_per_channel=[float(s) for s in pa_to_norm_per_ch],
                     would_clip_hardware=bool(would_clip),
                     mic_lo_hz=adc["mic_lo"], mic_hi_hz=adc["mic_hi"],
                     bit_depth=adc["bit_depth"])
    if achain["enabled"]:
        adc_meta["analog_chain_per_channel"] = achain_pc
        adc_meta["shared_adc_effects"] = mismatch_applied
    return final_signals, adc_meta


def write_wav(path, signals, fs):
    n = min(len(s) for s in signals)
    data = np.stack([s[:n] for s in signals], axis=1)
    clipped = bool(np.abs(data).max() > 1.0)
    data16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, fs, data16)
    return clipped


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: apply mic response, calibration, and ADC quantization to a Stage-2 noisy signal.")
    parser.add_argument("noisy_wav", help="Path to the *_noisy.wav from gs_gen_add_noise.py")
    parser.add_argument("config", nargs="?", default="gen_config.ini",
                         help="Path to .ini config file (default: gen_config.ini)")
    parser.add_argument("-o", "--output", default=None,
                         help="Output basename (overrides output.basename in config); "
                              "produces <output>.wav and <output>.json")
    parser.add_argument("--noisy-json", default=None,
                         help="Explicit path to the Stage-2 .json (auto-detected by default)")
    args = parser.parse_args()

    cfg = gp.load_config(args.config)
    if args.output:
        cfg["output"]["basename"] = args.output

    pa_signals, fs, noisy_meta = load_noisy_wav(args.noisy_wav, args.noisy_json)

    print(f"[Stage 3] Applying mic/ADC model to {args.noisy_wav} "
          f"({len(pa_signals)} channels, {fs} Hz)")

    final_signals, adc_meta = apply_adc(pa_signals, fs, cfg)

    wav_path = f"{cfg['output']['basename']}.wav"
    json_path = f"{cfg['output']['basename']}.json"

    clipped = write_wav(wav_path, final_signals, fs)
    clipped = clipped or adc_meta["would_clip_hardware"]   # quantize() clips internally,
                                                              # so write_wav's own check can
                                                              # never see genuine hardware
                                                              # overflow -- would_clip_hardware
                                                              # is computed BEFORE quantization
    if adc_meta["would_clip_hardware"]:
        print(f"WARNING: this hardware chain CLIPS on this signal -- peak "
              f"{adc_meta['peak_pa_before_calibration']:.2f} Pa exceeds what "
              f"{cfg['analog_chain']['preamp_gain_db']:.1f} dB gain into a "
              f"{cfg['analog_chain']['adc_vref_peak_v']:.2f} V ADC can represent. "
              f"Lower the preamp gain, use a less sensitive mic, or raise adc_vref_peak_v.")
    elif clipped:
        print(f"WARNING: signal clipped at full scale writing {wav_path} -- "
              f"consider raising calibration.real_peak_norm.")

    meta = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        stage="final_signal",
        wav_file=wav_path,
        source_noisy_wav=args.noisy_wav,
        sample_rate_hz=fs,
        n_channels=len(final_signals),
        n_samples=len(final_signals[0]),
        duration_s=len(final_signals[0])/fs,
        bit_depth_file=16,
        clipped=clipped,
        config_used=cfg,
        adc_processing=adc_meta,
        noise_processing=(noisy_meta["noise_processing"] if noisy_meta else None),
        clean_signal_ground_truth=(noisy_meta["clean_signal_ground_truth"] if noisy_meta else None),
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Stage 3] Wrote {wav_path}  ({len(final_signals)} channels, {fs} Hz, "
          f"{len(final_signals[0])/fs:.3f} s)")
    print(f"[Stage 3] Wrote {json_path}")


if __name__ == "__main__":
    main()
