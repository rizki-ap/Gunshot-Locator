#!/usr/bin/env python3
"""
generate_clean_signal.py
===========================
STAGE 1 of 2. Generates the pure, noiseless acoustic signal a gunshot
produces at each of the 4 microphones -- shockwave (N-wave) + muzzle blast
(Friedlander pulse), physics only. No noise, no reverb, no mic/ADC response,
no quantization. That's all added separately by add_noise.py.

Why separate: the physics here is deterministic given the config (geometry,
bullet, trajectory) -- run it once, then feed the same clean signal through
add_noise.py as many times as you like with different noise/reverb/ADC
settings, without re-deriving the physics each time.

Usage
-----
  python generate_clean_signal.py                      # uses config.ini in cwd
  python generate_clean_signal.py my_config.ini
  python generate_clean_signal.py my_config.ini -o my_shot

Output
------
  <output>_clean.wav   4-channel, 32-bit FLOAT PCM, actual Pa values (NOT
                        normalized to +-1 -- float WAV doesn't require that,
                        and normalizing here would lose the physical units
                        add_noise.py's calibration step needs)
  <output>_clean.json  ground-truth geometry/timing/amplitude for every mic

Only reads the [geometry], [bullet], [trajectory], [timing], and adc.fs
sections of the config -- everything noise/ADC-response/bit-depth related
belongs to Stage 2 and is ignored here.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.io import wavfile

import gunshot_physics as gp


def generate_clean(cfg):
    geo, bul, traj, tim = cfg["geometry"], cfg["bullet"], cfg["trajectory"], cfg["timing"]
    atm_cfg = cfg["atmosphere"]
    FS = cfg["adc"]["fs"]

    atmosphere = None
    if atm_cfg["enabled"]:
        atmosphere = dict(temp_c=atm_cfg["temp_c"], humidity_pct=atm_cfg["humidity_pct"],
                           pressure_kpa=atm_cfg["pressure_kpa"])

    # -- Section 1: array geometry --------------------------------------------
    mic_pos = gp.make_tetrahedron(geo["l_array"])
    n_mics = len(mic_pos)

    # -- Section 3: bullet property --------------------------------------------
    if bul["caliber"] not in gp.BULLET_LIBRARY:
        raise ValueError(
            f"Unknown bullet.caliber '{bul['caliber']}' -- valid options: "
            f"{list(gp.BULLET_LIBRARY.keys())}")
    bl = gp.BULLET_LIBRARY[bul["caliber"]]
    L_BULLET, dP0_sw, b0_sw = bl["L"], bl["dP0_sw"], bl["b0_sw"]
    P_REF_MB, R_REF_MB, T_POS_REF = bl["P_REF_MB"], bl["R_REF_MB"], bl["T_POS_REF"]
    M = bul["mach"]
    V_BULLET = M * gp.C

    # -- Section 4: trajectory / shooter position -------------------------------
    RANGE, Y_MISS = traj["range"], traj["y_miss"]
    BULLET_ORIGIN = np.array([-RANGE, Y_MISS, 0.0])
    V_HAT = np.array([1.0, 0.0, 0.0])
    SHOOTER_POS = BULLET_ORIGIN.copy()

    # -- Section 5.1: shockwave arrivals ----------------------------------------
    t_arr = np.zeros(n_mics); t_emi = np.zeros(n_mics)
    a_all = np.zeros(n_mics); b_all = np.zeros(n_mics)
    for i, p in enumerate(mic_pos):
        t_arr[i], t_emi[i], a_all[i], b_all[i] = gp.shockwave_arrival(
            p, BULLET_ORIGIN, V_HAT, M, gp.C)

    # -- Section 5.5: muzzle blast arrivals -------------------------------------
    r_mb_all = np.array([np.linalg.norm(p - SHOOTER_POS) for p in mic_pos])
    t_mb = r_mb_all / gp.C

    # -- Section 5.8: master timeline + ideal (noiseless) per-mic signal --------
    t0 = min(t_arr.min(), t_mb.min()) - tim["pre_roll"]
    t1 = t_mb.max() + tim["post_roll"]
    t_master = np.arange(t0, t1, 1.0/FS)

    T_Ns = np.zeros(n_mics); dP_sw_all = np.zeros(n_mics)
    t_pos_all = np.zeros(n_mics); dP_mb_all = np.zeros(n_mics)
    atten_sw_db = np.zeros(n_mics); atten_mb_db = np.zeros(n_mics)
    ideal_signals = []
    for i in range(n_mics):
        sw, T_N, dP_sw = gp.nwave_waveform(t_master, t_arr[i], b_all[i], M, gp.C,
                                            L_BULLET, b0_sw, dP0_sw, atmosphere=atmosphere)
        mb, t_pos, dP_mb = gp.friedlander_waveform(t_master, t_mb[i], r_mb_all[i],
                                                     P_REF_MB, R_REF_MB, T_POS_REF,
                                                     atmosphere=atmosphere)
        T_Ns[i], dP_sw_all[i], t_pos_all[i], dP_mb_all[i] = T_N, dP_sw, t_pos, dP_mb
        if atmosphere is not None:
            atten_sw_db[i] = gp.atmospheric_attenuation_db(1.0/T_N, b_all[i], **atmosphere)
            atten_mb_db[i] = gp.atmospheric_attenuation_db(1.0/t_pos, r_mb_all[i], **atmosphere)
        ideal_signals.append(sw + mb)

    ground_truth = dict(
        mic_pos=mic_pos.tolist(),
        shooter_pos=SHOOTER_POS.tolist(),
        bullet_origin=BULLET_ORIGIN.tolist(),
        trajectory_direction=V_HAT.tolist(),
        bullet_speed_mps=V_BULLET,
        sample_rate_hz=FS,
        atmosphere_applied=atmosphere,
        t_master_start_s=float(t0),
        t_master_end_s=float(t1),
        n_samples=len(t_master),
        per_mic=[
            dict(mic_index=i,
                 shockwave=dict(t_arrive_s=float(t_arr[i]), t_emit_s=float(t_emi[i]),
                                 along_track_a_m=float(a_all[i]), perp_dist_b_m=float(b_all[i]),
                                 duration_T_N_s=float(T_Ns[i]), peak_dP_pa=float(dP_sw_all[i]),
                                 atmospheric_attenuation_db=float(atten_sw_db[i])),
                 muzzle_blast=dict(t_arrive_s=float(t_mb[i]), range_m=float(r_mb_all[i]),
                                    positive_phase_t_pos_s=float(t_pos_all[i]),
                                    peak_dP_pa=float(dP_mb_all[i]),
                                    atmospheric_attenuation_db=float(atten_mb_db[i])))
            for i in range(n_mics)
        ],
        true_tdoa_sw_us=[float((t_arr[i]-t_arr[0])*1e6) for i in range(1, n_mics)],
        true_tdoa_mb_us=[float((t_mb[i]-t_mb[0])*1e6) for i in range(1, n_mics)],
        dt_mb_minus_sw_centroid_s=float(t_mb.mean() - t_arr.mean()),
    )
    return ideal_signals, FS, ground_truth


def write_clean_wav(path, signals, fs):
    """Float32 PCM, actual Pa values (unnormalized) -- lossless handoff to
    Stage 2. Most audio tools can play float WAVs; the values just won't be
    in the usual +-1 convention, which is intentional here."""
    n = min(len(s) for s in signals)
    data = np.stack([s[:n] for s in signals], axis=1).astype(np.float32)
    wavfile.write(path, fs, data)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: generate the noiseless 4-mic gunshot signal (physics only).")
    parser.add_argument("config", nargs="?", default="config.ini",
                         help="Path to .ini config file (default: config.ini)")
    parser.add_argument("-o", "--output", default=None,
                         help="Output basename (overrides output.basename in config); "
                              "produces <output>_clean.wav and <output>_clean.json")
    args = parser.parse_args()

    cfg = gp.load_config(args.config)
    if args.output:
        cfg["output"]["basename"] = args.output

    print(f"[Stage 1] Generating clean signal: caliber={cfg['bullet']['caliber']}, "
          f"Mach={cfg['bullet']['mach']}, range={cfg['trajectory']['range']} m")

    signals, fs, ground_truth = generate_clean(cfg)

    wav_path = f"{cfg['output']['basename']}_clean.wav"
    json_path = f"{cfg['output']['basename']}_clean.json"

    write_clean_wav(wav_path, signals, fs)
    with open(json_path, "w") as f:
        json.dump(dict(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            stage="clean_signal",
            wav_file=wav_path,
            wav_format="float32 PCM, unnormalized Pa units",
            sample_rate_hz=fs,
            n_channels=len(signals),
            n_samples=len(signals[0]),
            duration_s=len(signals[0])/fs,
            config_used=cfg,
            ground_truth=ground_truth,
        ), f, indent=2)

    print(f"[Stage 1] Wrote {wav_path}  ({len(signals)} channels, {fs} Hz, "
          f"{len(signals[0])/fs:.3f} s, noiseless)")
    print(f"[Stage 1] Wrote {json_path}")
    print(f"[Stage 1] Next: python add_noise.py {wav_path} {args.config}")


if __name__ == "__main__":
    main()
