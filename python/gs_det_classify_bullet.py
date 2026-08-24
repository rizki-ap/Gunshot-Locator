#!/usr/bin/env python3
"""
gs_det_classify_bullet.py
============================
DETECTION STAGE 4. Reads Stage 2's TDOA output and estimates the bullet's
Mach number and caliber via three independent methods (adapted from the
validated Section 9 velocity-estimation approach):

  Method A -- angle only: k_hat_sw . v_hat ~ 1/M, using v_hat ~ -n_hat_shooter
    as a cheap approximation of bullet direction. No amplitude calibration
    needed, but least accurate (biased by miss distance).
  Method B -- duration + amplitude: solves the Whitham N-wave duration and
    amplitude equations jointly for M, for EACH caliber in the reference
    library. Needs this detector's own amplitude calibration (see
    det_config.ini [calibration]) to convert normalized dP_sw back to Pa.
  Method C -- geometric correction: uses Method B's miss-distance estimate
    and a self-contained PILAR-V range solve to correct the approximate
    bullet direction, rather than assuming v_hat ~ -n_hat_shooter directly.

Caliber classification: Method B is run against EVERY caliber in the
reference library; the caliber whose fit gives the most physically self-
consistent (real, supersonic) Mach solution is reported as the best guess.
This can't be highly confident with only 2 reference calibers that share
most calibration constants (see det_config.ini) -- treat it as a
demonstration of the approach, not a rigorously validated classifier.

Usage
-----
  python gs_det_classify_bullet.py det_result_tdoa.json                # det_config.ini
  python gs_det_classify_bullet.py det_result_tdoa.json my_det_config.ini
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq

from gs_det_signal_prepare import load_det_config, make_tetrahedron
from gs_det_shooter_locator import solve_doa, pilar_v_exact


def method_a_angle_only(k_hat_sw, n_hat_shooter):
    v_hat_approx = -n_hat_shooter
    dot = float(np.dot(k_hat_sw, v_hat_approx))
    return 1.0 / dot if dot > 0 else None


def method_b_duration_amplitude(T_N, dP, c, bl):
    """Solve Whitham's T_N(M,b) and dP(b) equations jointly for M, for one
    candidate caliber's L/b0_sw/dP0_sw. Returns (M, b) or None if no valid
    supersonic root exists for this caliber."""
    if T_N is None or np.isnan(T_N) or T_N <= 0 or dP is None or dP <= 0:
        return None
    b_est = bl["b0_sw"] * (bl["dP0_sw"] / dP) ** 2

    def f_M(M):
        if M <= 1.0:
            return -1e10
        return T_N ** 2 * M ** 2 * c ** 2 * np.sqrt(M ** 2 - 1) - 4.0 * b_est * bl["L"]

    if f_M(1.001) * f_M(10.0) > 0:
        return None
    M = brentq(f_M, 1.001, 10.0)
    return M, b_est


def method_c_geometric_correction(k_hat_sw, n_hat_shooter, b_corr, M_B, dt_sw_to_mb, c):
    """Refine v_hat using a self-contained PILAR-V range solve (independent
    of gs_det_shooter_locator.py's own range estimate -- this is its own
    internal solve, matching how the notebook's Method C was self-contained
    too)."""
    if dt_sw_to_mb is None or M_B is None or M_B <= 1.0:
        return None
    try:
        r_est = pilar_v_exact(dt_sw_to_mb, b_corr, M_B, c)
    except (ValueError, RuntimeError):
        return None
    sin_alpha = b_corr / np.sqrt(r_est ** 2 + b_corr ** 2)
    cos_alpha = np.sqrt(max(0.0, 1 - sin_alpha ** 2))
    k_sw_along = np.dot(k_hat_sw, n_hat_shooter) * n_hat_shooter
    e_lat = -(k_hat_sw - k_sw_along)
    norm = np.linalg.norm(e_lat)
    if norm < 1e-9:
        return None
    e_lat = e_lat / norm
    v_hat_corr = cos_alpha * (-n_hat_shooter) + sin_alpha * e_lat
    v_hat_corr /= np.linalg.norm(v_hat_corr)
    dot_C = float(np.dot(k_hat_sw, v_hat_corr))
    return 1.0 / dot_C if dot_C > 0 else None


def main():
    parser = argparse.ArgumentParser(
        description="Detection Stage 4: bullet Mach/caliber classification (3 methods).")
    parser.add_argument("tdoa_json", help="Path to Stage 2's *_tdoa.json")
    parser.add_argument("config", nargs="?", default="det_config.ini",
                         help="Path to detection .ini config (default: det_config.ini)")
    parser.add_argument("-o", "--output", default=None,
                         help="Output basename (overrides output.basename in config)")
    args = parser.parse_args()

    cfg = load_det_config(args.config)
    if args.output:
        cfg["output"]["basename"] = args.output

    with open(args.tdoa_json) as f:
        tdoa_data = json.load(f)

    mic_pos = make_tetrahedron(cfg["geometry"]["l_array"])
    c = cfg["physics"]["speed_of_sound"]
    ref_ch = tdoa_data["reference_channel"]

    k_hat_sw, _ = solve_doa(mic_pos, tdoa_data["tdoa_sw_s"], c, ref_ch)

    result = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        stage="bullet_classification",
        source_tdoa_json=args.tdoa_json,
    )

    if not tdoa_data["mb_available"]:
        print("[Det Stage 4] No muzzle-blast TDOA available -- classification needs bearing, "
              "which needs MB TDOA. Aborting.")
        result["classification_available"] = False
        with open(f"{cfg['output']['basename']}_classification.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    n_hat_shooter, _ = solve_doa(mic_pos, tdoa_data["tdoa_mb_s"], c, ref_ch)
    n_hat_shooter = -n_hat_shooter

    T_N = tdoa_data.get("t_n_sw_ref_channel")
    dp_sw_norm = tdoa_data.get("dp_sw_ref_channel")
    dp_sw_pa = (abs(dp_sw_norm) * cfg["calibration"]["assumed_peak_pa_at_full_scale"]
                if dp_sw_norm is not None else None)
    dt_sw_to_mb = tdoa_data.get("dt_sw_to_mb_s")

    M_A = method_a_angle_only(k_hat_sw, n_hat_shooter)
    result["method_a_mach"] = M_A
    print(f"[Det Stage 4] Method A (angle only): Mach = {M_A}")

    print(f"[Det Stage 4] Testing {len(cfg['bullet_reference_library'])} candidate caliber(s) "
          f"against Method B (T_N={T_N}, dP={dp_sw_pa} Pa assumed)...")
    caliber_results = {}
    for cal, bl in cfg["bullet_reference_library"].items():
        mb_result = method_b_duration_amplitude(T_N, dp_sw_pa, c, bl)
        if mb_result is None:
            caliber_results[cal] = dict(mach=None, miss_distance_m=None, valid=False)
            print(f"    {cal}: no valid supersonic solution")
            continue
        M_B, b_corr = mb_result
        M_C = method_c_geometric_correction(k_hat_sw, n_hat_shooter, b_corr, M_B, dt_sw_to_mb, c)
        caliber_results[cal] = dict(mach_method_b=M_B, mach_method_c=M_C,
                                     miss_distance_m=b_corr, valid=True)
        print(f"    {cal}: Method B Mach = {M_B:.3f}, Method C Mach = {M_C}, "
              f"b = {b_corr:.2f} m")

    result["caliber_candidates"] = caliber_results

    valid_calibers = {k: v for k, v in caliber_results.items() if v["valid"]}
    if valid_calibers:
        # Best guess: smallest |M - typical supersonic midpoint| as a simple,
        # transparent tie-break -- with only 2 near-identical reference
        # calibers in this library, don't oversell this as a validated
        # classifier (see module docstring).
        best_cal = min(valid_calibers, key=lambda k: abs(valid_calibers[k]["mach_method_b"] - 2.25))
        result["best_guess_caliber"] = best_cal
        print(f"[Det Stage 4] Best-guess caliber: {best_cal}")
    else:
        result["best_guess_caliber"] = None
        print("[Det Stage 4] No caliber produced a valid solution.")

    out_json = f"{cfg['output']['basename']}_classification.json"
    result["classification_available"] = True
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Det Stage 4] Wrote {out_json}")


if __name__ == "__main__":
    main()
