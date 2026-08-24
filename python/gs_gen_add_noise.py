#!/usr/bin/env python3
"""
gs_gen_add_noise.py
==============
STAGE 2 of 3. Reads the noiseless per-mic signal written by
gs_gen_clean_signal.py and adds ENVIRONMENTAL noise/reverb only --
nothing hardware-specific. Mic frequency response, ADC bit-depth
quantization, and sensitivity calibration are Stage 3's job (gs_gen_apply_adc.py),
not this one: those are properties of your sensor/ADC hardware, not of the
environment the shot was fired in, and keeping them separate means you can
mix-and-match freely (same noisy environment through different ADCs, or the
same ADC across different environments) without re-deriving anything.

Run gs_gen_clean_signal.py ONCE, then run this as many times as you like
with different [noise] settings -- the physics doesn't need to be re-derived
each time, only the environmental noise model.

Usage
-----
  python gs_gen_add_noise.py gunshot_sim_clean.wav                     # gen_config.ini
  python gs_gen_add_noise.py gunshot_sim_clean.wav my_config.ini
  python gs_gen_add_noise.py gunshot_sim_clean.wav my_config.ini -o my_shot

Output
------
  <output>_noisy.wav   4-channel, 32-bit FLOAT PCM, still in Pa units (like
                        Stage 1's output -- NOT yet mic-filtered, calibrated,
                        or quantized, so precision is preserved for Stage 3)
  <output>_noisy.json  clean-signal ground truth + noise-processing details

Only reads the [noise] and [output] sections of the config -- [adc] and
[calibration] belong to Stage 3.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

import gs_gen_physic as gp


def load_clean_wav(wav_path, json_path=None):
    """Load Stage 1's float32 clean .wav (+ its .json for metadata passthrough,
    if present -- looked up automatically by convention if not given)."""
    fs, data = wavfile.read(wav_path)
    if data.dtype != np.float32:
        print(f"WARNING: expected float32 clean .wav, got {data.dtype}. "
              f"Proceeding, but check this is really a Stage-1 output.")
    signals = [data[:, i].astype(np.float64) for i in range(data.shape[1])]

    if json_path is None:
        json_path = wav_path.rsplit(".wav", 1)[0] + ".json"
        if not json_path.endswith("_clean.json") and wav_path.endswith("_clean.wav"):
            json_path = wav_path[:-len("_clean.wav")] + "_clean.json"
    clean_meta = None
    try:
        with open(json_path) as f:
            clean_meta = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: no companion metadata file found at '{json_path}' -- "
              f"output .json will be missing ground-truth physics info.")

    return signals, fs, clean_meta


def apply_noise(ideal_signals, t_master_len, fs, cfg):
    """Environmental noise/reverb ONLY. Output stays in Pa units, float --
    no mic response, no calibration, no quantization (that's Stage 3)."""
    noi, out = cfg["noise"], cfg["output"]
    n_mics = len(ideal_signals)
    seed = out["seed"]

    if noi["model"] == "simple":
        rng_adc = np.random.default_rng(seed)
        pa_signals = [ideal_signals[i] + noi["noise_floor_pa"]*rng_adc.standard_normal(t_master_len)
                      for i in range(n_mics)]
    elif noi["model"] == "realistic":
        pa_signals = []
        for i in range(n_mics):
            dry = ideal_signals[i]
            rir = gp.make_room_ir(noi["rt60"], fs, seed=1000+seed+i)
            wet = fftconvolve(dry, rir, mode="full")[:len(dry)]
            if np.abs(dry).max() > 0:
                wet *= np.abs(dry).max() / (np.abs(wet).max() + 1e-12)
            noise = gp.make_colored_noise(len(dry), fs, noi["noise_slope"], noi["noise_rms_pa"],
                                           seed=2000+seed+i)
            pa_signals.append(wet + noise)
    else:
        raise ValueError(f"Unknown noise.model '{noi['model']}' -- use 'simple' or 'realistic'.")

    noise_meta = dict(model=noi["model"])
    return pa_signals, noise_meta


def write_noisy_wav(path, signals, fs):
    """Float32 PCM, still in Pa units -- same lossless-handoff convention as
    Stage 1's clean .wav, so Stage 3 gets full precision to work with."""
    n = min(len(s) for s in signals)
    data = np.stack([s[:n] for s in signals], axis=1).astype(np.float32)
    wavfile.write(path, fs, data)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: add environmental noise/reverb to a Stage-1 clean signal.")
    parser.add_argument("clean_wav", help="Path to the *_clean.wav from gs_gen_clean_signal.py")
    parser.add_argument("config", nargs="?", default="gen_config.ini",
                         help="Path to .ini config file (default: gen_config.ini)")
    parser.add_argument("-o", "--output", default=None,
                         help="Output basename (overrides output.basename in config); "
                              "produces <output>_noisy.wav and <output>_noisy.json")
    parser.add_argument("--clean-json", default=None,
                         help="Explicit path to the Stage-1 .json (auto-detected by default)")
    args = parser.parse_args()

    cfg = gp.load_config(args.config)
    if args.output:
        cfg["output"]["basename"] = args.output

    ideal_signals, fs, clean_meta = load_clean_wav(args.clean_wav, args.clean_json)

    print(f"[Stage 2] Adding noise model '{cfg['noise']['model']}' to {args.clean_wav} "
          f"({len(ideal_signals)} channels, {fs} Hz)")

    pa_signals, noise_meta = apply_noise(ideal_signals, len(ideal_signals[0]), fs, cfg)

    wav_path = f"{cfg['output']['basename']}_noisy.wav"
    json_path = f"{cfg['output']['basename']}_noisy.json"

    write_noisy_wav(wav_path, pa_signals, fs)

    meta = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        stage="noisy_signal",
        wav_file=wav_path,
        wav_format="float32 PCM, unnormalized Pa units (mic/ADC not yet applied)",
        source_clean_wav=args.clean_wav,
        sample_rate_hz=fs,
        n_channels=len(pa_signals),
        n_samples=len(pa_signals[0]),
        duration_s=len(pa_signals[0])/fs,
        config_used=cfg,
        noise_processing=noise_meta,
        clean_signal_ground_truth=(clean_meta["ground_truth"] if clean_meta else None),
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Stage 2] Wrote {wav_path}  ({len(pa_signals)} channels, {fs} Hz, "
          f"{len(pa_signals[0])/fs:.3f} s, still Pa units)")
    print(f"[Stage 2] Wrote {json_path}")
    print(f"[Stage 2] Next: python gs_gen_apply_adc.py {wav_path} {args.config}")


if __name__ == "__main__":
    main()

