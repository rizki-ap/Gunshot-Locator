"""
gs_gen_physic.py
====================
Shared physics + config-loading library for the two-stage gunshot signal
pipeline:

  gs_gen_clean_signal.py   ->  <basename>_clean.wav + <basename>_clean.json
  gs_gen_add_noise.py                ->  <basename>.wav       + <basename>.json

All physics here is unchanged from gunshot_generate_detect.ipynb (Sections
1, 3, 4, 5.1, 5.3, 5.6, 5.8, 6.1, 7) -- this module just holds the shared
code so the two stages don't duplicate (and risk diverging on) the same
math. Not meant to be run directly.
"""

import configparser
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve


# ===========================================================================
# Physical constants
# ===========================================================================
C = 343.0             # Speed of sound (m/s), ISA sea level
P_ATM = 101325.0       # Atmospheric pressure (Pa)

BULLET_LIBRARY = {
    '7.62_NATO': dict(L=0.028, dP0_sw=7.5, b0_sw=50.0,
                       P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003),
    '5.56_NATO': dict(L=0.023, dP0_sw=7.5, b0_sw=50.0,
                       P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003/4.0),
}


# ===========================================================================
# Config loading -- shared schema, each stage only reads the sections it needs
# ===========================================================================
DEFAULTS = {
    "geometry":    {"l_array": "0.30"},
    "adc":         {"fs": "100000", "bit_depth": "16",
                     "mic_lo": "40.0", "mic_hi": "20000.0"},
    "calibration": {"real_noise_rms": "0.000595", "real_rt60": "1.32",
                     "real_peak_norm": "0.098", "real_noise_slope": "-2.46"},
    "analog_chain": {"enabled": "false", "mic_sensitivity_mv_per_pa": "22.4",
                      "preamp_gain_db": "40.0", "adc_vref_peak_v": "2.5"},
    "adc_multi_channel": {"type": "simultaneous", "conversion_time_us": "2.0",
                           "gain_mismatch_pct": "0.0", "offset_mismatch_mv": "0.0",
                           "mismatch_seed": "42"},
    "bullet":      {"caliber": "5.56_NATO", "mach": "2.5"},
    "trajectory":  {"range": "200.0", "y_miss": "10.0"},
    "atmosphere":  {"enabled": "true", "temp_c": "20.0", "humidity_pct": "50.0",
                     "pressure_kpa": "101.325"},
    "timing":      {"pre_roll": "0.010", "post_roll": "0.150"},
    "noise":       {"model": "realistic", "noise_floor_pa": "0.005",
                     "rt60": "1.25", "noise_rms_pa": "0.075", "noise_slope": "-2.4"},
    "output":      {"basename": "gunshot_sim", "seed": "0"},
}


def load_config(path):
    """Load an .ini config file, falling back to DEFAULTS for anything
    missing (including a missing file entirely). Returns a nested dict of
    RESOLVED values (native types) -- used for computation AND echoed back
    verbatim into the output .json for reproducibility.

    Per-channel hardware: optional [analog_chain_ch0]..[analog_chain_ch3]
    sections override mic_sensitivity_mv_per_pa/preamp_gain_db for that
    specific channel; anything not overridden falls back to [analog_chain]'s
    nominal values. Result is in cfg["analog_chain_per_channel"], a list of
    4 dicts. This project's array is fixed at 4 mics (make_tetrahedron
    always returns 4 points), so channels 0-3 are hardcoded here."""
    cp = configparser.ConfigParser()
    cp.read_dict(DEFAULTS)
    if path is not None:
        found = cp.read(path)
        if not found:
            print(f"WARNING: config file '{path}' not found -- using built-in defaults.",
                  file=sys.stderr)

    n_channels = 4
    nominal = dict(mic_sensitivity_mv_per_pa=cp.getfloat("analog_chain", "mic_sensitivity_mv_per_pa"),
                   preamp_gain_db=cp.getfloat("analog_chain", "preamp_gain_db"))
    per_channel = []
    for i in range(n_channels):
        section = f"analog_chain_ch{i}"
        ch = dict(nominal)
        if cp.has_section(section):
            if cp.has_option(section, "mic_sensitivity_mv_per_pa"):
                ch["mic_sensitivity_mv_per_pa"] = cp.getfloat(section, "mic_sensitivity_mv_per_pa")
            if cp.has_option(section, "preamp_gain_db"):
                ch["preamp_gain_db"] = cp.getfloat(section, "preamp_gain_db")
        per_channel.append(ch)

    return {
        "geometry": dict(l_array=cp.getfloat("geometry", "l_array")),
        "adc": dict(fs=cp.getint("adc", "fs"),
                    bit_depth=cp.getint("adc", "bit_depth"),
                    mic_lo=cp.getfloat("adc", "mic_lo"),
                    mic_hi=cp.getfloat("adc", "mic_hi")),
        "calibration": dict(real_noise_rms=cp.getfloat("calibration", "real_noise_rms"),
                             real_rt60=cp.getfloat("calibration", "real_rt60"),
                             real_peak_norm=cp.getfloat("calibration", "real_peak_norm"),
                             real_noise_slope=cp.getfloat("calibration", "real_noise_slope")),
        "analog_chain": dict(enabled=cp.getboolean("analog_chain", "enabled"),
                              mic_sensitivity_mv_per_pa=nominal["mic_sensitivity_mv_per_pa"],
                              preamp_gain_db=nominal["preamp_gain_db"],
                              adc_vref_peak_v=cp.getfloat("analog_chain", "adc_vref_peak_v")),
        "analog_chain_per_channel": per_channel,
        "adc_multi_channel": dict(
            type=cp.get("adc_multi_channel", "type"),
            conversion_time_us=cp.getfloat("adc_multi_channel", "conversion_time_us"),
            gain_mismatch_pct=cp.getfloat("adc_multi_channel", "gain_mismatch_pct"),
            offset_mismatch_mv=cp.getfloat("adc_multi_channel", "offset_mismatch_mv"),
            mismatch_seed=cp.getint("adc_multi_channel", "mismatch_seed")),
        "bullet": dict(caliber=cp.get("bullet", "caliber"),
                       mach=cp.getfloat("bullet", "mach")),
        "trajectory": dict(range=cp.getfloat("trajectory", "range"),
                            y_miss=cp.getfloat("trajectory", "y_miss")),
        "atmosphere": dict(enabled=cp.getboolean("atmosphere", "enabled"),
                            temp_c=cp.getfloat("atmosphere", "temp_c"),
                            humidity_pct=cp.getfloat("atmosphere", "humidity_pct"),
                            pressure_kpa=cp.getfloat("atmosphere", "pressure_kpa")),
        "timing": dict(pre_roll=cp.getfloat("timing", "pre_roll"),
                        post_roll=cp.getfloat("timing", "post_roll")),
        "noise": dict(model=cp.get("noise", "model"),
                      noise_floor_pa=cp.getfloat("noise", "noise_floor_pa"),
                      rt60=cp.getfloat("noise", "rt60"),
                      noise_rms_pa=cp.getfloat("noise", "noise_rms_pa"),
                      noise_slope=cp.getfloat("noise", "noise_slope")),
        "output": dict(basename=cp.get("output", "basename"),
                       seed=cp.getint("output", "seed")),
    }


# ===========================================================================
# Section 1 -- Sensor geometry
# ===========================================================================
def make_tetrahedron(edge_length):
    raw = np.array([[0.000, 0.000, 1.000], [0.000, 0.943, -0.333],
                     [-0.816, -0.471, -0.333], [0.816, -0.471, -0.333]], dtype=float)
    scale = edge_length / np.linalg.norm(raw[0] - raw[1])
    return raw * scale


# ===========================================================================
# Section 2 -- Mic sensitivity / ADC (used only by the noise-adding stage)
# ===========================================================================
def mic_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, min(hi, fs/2*0.99)], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def quantize(x, bits, full_scale=1.0):
    levels = 2 ** (bits - 1)
    return np.round(np.clip(x, -full_scale, full_scale) * levels) / levels


# ===========================================================================
# Analog gain chain (mic sensitivity -> preamp gain -> ADC reference) --
# an alternative, hardware-derived way to compute the Pa-to-normalized-code
# calibration scale, instead of empirically peak-matching to a real
# recording. Useful for evaluating a candidate mic/preamp/ADC combination
# from datasheet numbers alone, before you have a real recording with that
# hardware to calibrate against.
#
# Chain:  v_mic = S_mic . p          (mic sensitivity, V/Pa)
#         v_out = 10^(G_dB/20) . v_mic   (preamp gain)
#         code  = v_out / V_ref_peak      (normalized to +-1 at full scale)
#
#   => PA_TO_NORM = S_mic . 10^(G_dB/20) / V_ref_peak
# ===========================================================================
def dbv_per_pa_to_v_per_pa(dbv_per_pa):
    """Convert a mic sensitivity given in dBV/Pa (a common datasheet
    convention, e.g. '-38 dBV/Pa' or 'dBV re 1V/Pa at 94dB SPL / 1kHz') to
    linear V/Pa."""
    return 10 ** (dbv_per_pa / 20.0)


def gain_chain_scale(mic_sensitivity_v_per_pa, preamp_gain_db, adc_vref_peak_v):
    """PA_TO_NORM scale factor from explicit hardware specs: mic sensitivity
    (V/Pa), preamp gain (dB), and the ADC's peak input voltage for full
    scale (the voltage that maps to a normalized code of +-1.0 -- check your
    ADC's datasheet for whether this is Vref, Vref/2, or something else
    depending on its input architecture)."""
    preamp_gain_linear = 10 ** (preamp_gain_db / 20.0)
    return mic_sensitivity_v_per_pa * preamp_gain_linear / adc_vref_peak_v


# ===========================================================================
# Shared-ADC, multi-channel effects: what a SINGLE ADC chip sampling all 4
# mic channels adds beyond each channel's own mic/preamp gain-chain scale.
#
#   - Channel-to-channel gain/offset mismatch: real, but usually a small
#     effect (~0.1-0.5% typical datasheet spec).
#   - Timing skew, IF the ADC is multiplexed rather than simultaneous-
#     sampling: NOT a small effect for a TDOA-based system -- a multiplexed
#     ADC samples channels sequentially, so channel i is delayed by
#     i * conversion_time relative to channel 0, directly corrupting the
#     microsecond-scale timing this whole array exists to measure.
# ===========================================================================
def fractional_delay(x, delay_samples):
    """Delay a signal by a (possibly non-integer) number of samples, via
    FFT-domain linear phase shift -- exact for band-limited signals, unlike
    a simple np.roll which only handles whole-sample shifts. Validated
    against np.roll for integer delays (agreement to float64 precision) and
    against cross-correlation-estimated sub-sample lag for fractional
    delays (agreement to <0.001 samples)."""
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    phase_shift = np.exp(-2j * np.pi * freqs * delay_samples)
    return np.fft.irfft(X * phase_shift, n=n)


# ===========================================================================
# Section 5.1 -- Shockwave (Mach cone) arrival physics
# ===========================================================================
def shockwave_arrival(sensor_pos, bullet_origin, v_hat, M, c):
    beta = np.sqrt(M**2 - 1.0)
    r = sensor_pos - bullet_origin
    a = float(np.dot(r, v_hat))
    b = float(np.linalg.norm(r - a * v_hat))
    if a - b / beta < 0:
        raise ValueError(
            f"Sensor outside Mach cone (a={a:.1f} < b/beta={b/beta:.1f}). "
            "Increase trajectory.range or reduce trajectory.y_miss.")
    return (a + b*beta)/(M*c), (a - b/beta)/(M*c), a, b   # t_arrive, t_emit, a, b


# ===========================================================================
# Section 5.3 -- N-wave (Whitham weak-shock model)
# ===========================================================================
def nwave_duration(b, M, c, L):
    return (2.0/(M*c)) * np.sqrt(b*L / np.sqrt(M**2 - 1.0))


def nwave_peak_pressure(b, b_ref, dP_ref, freq_hz=None, distance_m=None, atmosphere=None):
    dP = dP_ref * np.sqrt(b_ref/b)
    if atmosphere is not None and freq_hz is not None and distance_m is not None:
        dP, _ = apply_atmospheric_attenuation(dP, freq_hz, distance_m, atmosphere)
    return dP


def nwave_waveform(t, t_arrive, b, M, c, L, b_ref, dP_ref, atmosphere=None):
    """atmosphere: None (default, no absorption -- original behavior) or an
    atmosphere-params dict (see atmospheric_absorption_coefficient). When
    given, dP is additionally attenuated using the N-wave's characteristic
    frequency (~1/T_N) over a propagation distance of b (the shockwave
    reaches the sensor from its closest approach on the trajectory, roughly
    distance b away -- consistent with the existing b-based spreading law)."""
    T_N = nwave_duration(b, M, c, L)
    f_char = 1.0 / T_N
    dP = nwave_peak_pressure(b, b_ref, dP_ref, freq_hz=f_char, distance_m=b, atmosphere=atmosphere)
    p = np.zeros(len(t))
    mask = (t >= t_arrive) & (t < t_arrive + T_N)
    tau = t[mask] - t_arrive
    p[mask] = dP * (1.0 - 2.0*tau/T_N)
    return p, T_N, dP


# ===========================================================================
# Section 5.6 -- Friedlander muzzle blast (Hopkinson-Cranz scaling)
# ===========================================================================
def friedlander_params(r, P_REF_MB, R_REF_MB, T_POS_REF, atmosphere=None):
    dP = P_REF_MB * (R_REF_MB/r)
    t_pos = T_POS_REF * (r/R_REF_MB) ** (1.0/3.0)
    if atmosphere is not None:
        f_char = 1.0 / t_pos
        dP, _ = apply_atmospheric_attenuation(dP, f_char, r, atmosphere)
    return dP, t_pos


def friedlander_waveform(t, t_arrive, r, P_REF_MB, R_REF_MB, T_POS_REF, atmosphere=None):
    """atmosphere: None (default) or an atmosphere-params dict. Muzzle blast
    propagates spherically over the full range r, so both the geometric
    (1/r) and atmospheric attenuation use the same distance r."""
    dP, t_pos = friedlander_params(r, P_REF_MB, R_REF_MB, T_POS_REF, atmosphere=atmosphere)
    p = np.zeros(len(t))
    mask = t >= t_arrive
    tau = t[mask] - t_arrive
    p[mask] = dP * (1.0 - tau/t_pos) * np.exp(-tau/t_pos)
    return p, t_pos, dP


# ===========================================================================
# Atmospheric absorption (ISO 9613-1) -- optional, additional to the
# existing geometric (1/sqrt(b) or 1/r) spreading laws above. Frequency-
# dependent molecular absorption from oxygen/nitrogen relaxation, the
# dominant loss mechanism at ranges beyond a few hundred meters that the
# pure geometric-spreading model has no way to capture on its own.
#
# Validated against published reference values (ISO 9613-1 / Bass et al.
# 1995 style tables): at 4000 Hz, 20 C, this implementation computes
# 109.8 dB/km at 10% RH (reference: 109 dB/km) and 23.1 dB/km at 70% RH
# (reference: 23 dB/km) -- both within ~1%.
# ===========================================================================
DEFAULT_ATMOSPHERE = dict(temp_c=20.0, humidity_pct=50.0, pressure_kpa=101.325)


def atmospheric_absorption_coefficient(freq_hz, temp_c=20.0, humidity_pct=50.0, pressure_kpa=101.325):
    """ISO 9613-1 atmospheric absorption coefficient, in dB/m, for a pure
    tone at freq_hz. Combines classical (viscous/thermal) absorption with
    oxygen and nitrogen molecular relaxation, both humidity- and
    temperature-dependent."""
    T = temp_c + 273.15       # K
    T0 = 293.15                # reference temp, 20 C
    T01 = 273.16                # triple-point temp
    Pr = 101.325                # reference pressure, kPa
    Pa = pressure_kpa
    hr = humidity_pct
    f = np.asarray(freq_hz, dtype=float)

    psat_over_pr = 10 ** (-6.8346 * (T01/T)**1.261 + 4.6151)
    h = hr * psat_over_pr       # molar concentration of water vapor, %

    frO = (Pa/Pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (Pa/Pr) * (T/T0)**(-0.5) * (9 + 280*h*np.exp(-4.170 * ((T/T0)**(-1.0/3.0) - 1)))

    term1 = 1.84e-11 * (Pr/Pa) * (T/T0)**0.5
    term2 = (T/T0)**(-2.5) * (
        0.01275 * np.exp(-2239.1/T) / (frO + f**2/frO) +
        0.1068 * np.exp(-3352.0/T) / (frN + f**2/frN)
    )
    return 8.686 * f**2 * (term1 + term2)   # dB/m


def atmospheric_attenuation_db(freq_hz, distance_m, temp_c=20.0, humidity_pct=50.0, pressure_kpa=101.325):
    return atmospheric_absorption_coefficient(freq_hz, temp_c, humidity_pct, pressure_kpa) * distance_m


def apply_atmospheric_attenuation(dP, freq_hz, distance_m, atmosphere):
    """Reduce a peak-pressure value dP by ISO 9613-1 atmospheric absorption
    at freq_hz over distance_m. atmosphere: dict with temp_c/humidity_pct/
    pressure_kpa (missing keys fall back to DEFAULT_ATMOSPHERE). Returns
    (dP_attenuated, attenuation_db)."""
    params = dict(DEFAULT_ATMOSPHERE)
    params.update(atmosphere)
    atten_db = atmospheric_attenuation_db(freq_hz, distance_m, **params)
    return dP * 10**(-atten_db/20.0), float(atten_db)


# ===========================================================================
# Section 6.1 -- Noise / reverberation models (used only by gs_gen_add_noise.py)
# ===========================================================================
def make_room_ir(rt60, fs, duration=None, seed=None):
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = rt60 * 1.2
    n = int(duration * fs)
    t = np.arange(n) / fs
    decay = 10 ** (-3.0*t/rt60)
    ir = decay * rng.standard_normal(n)
    ir[0] = 1.0
    return ir


def make_colored_noise(n, fs, slope, rms, seed=None):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    freqs[0] = freqs[1]
    X = X * (freqs ** (slope/2.0))
    colored = np.fft.irfft(X, n=n)
    return colored / (np.sqrt(np.mean(colored**2)) + 1e-12) * rms
