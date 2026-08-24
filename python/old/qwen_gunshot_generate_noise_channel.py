import numpy as np
import json
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION & HELPER FUNCTIONS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. ACOUSTIC ENVIRONMENT NOISE MODELS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def make_statistical_rir(rt60, fs, n_early=4):
    """Statistical Room Impulse Response: exponentially-decaying noise + early reflections."""
    rng = np.random.default_rng()
    duration = rt60 * 1.2
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Late reverberation tail
    decay = 10 ** (-3.0 * t / rt60)
    late_rev = decay * rng.standard_normal(n_samples)
    
    # Early discrete reflections
    rir = np.zeros(n_samples)
    rir[0] = 1.0  # Direct sound (0 ms)
    for _ in range(n_early):
        delay_ms = rng.uniform(2.0, 30.0)
        delay_samp = int(delay_ms * 1e-3 * fs)
        atten = rng.uniform(0.1, 0.6)
        polarity = rng.choice([-1.0, 1.0])
        if delay_samp < n_samples:
            rir[delay_samp] += atten * polarity
            
    rir += late_rev
    rir /= np.max(np.abs(rir))
    return rir

def make_colored_noise(n_samples, fs, slope, rms_pa):
    """Noise with a target PSD slope (0=white, -1=pink, -2=brown) via FFT filtering."""
    rng = np.random.default_rng()
    white = rng.standard_normal(n_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    freqs[0] = freqs[1]  # Avoid divide-by-zero at DC
    
    X = X * (freqs ** (slope / 2.0))
    colored = np.fft.irfft(X, n=n_samples)
    
    current_rms = np.sqrt(np.mean(colored**2))
    if current_rms > 0:
        colored = colored * (rms_pa / current_rms)
    return colored

def make_impulsive_noise(n_samples, fs, rate_hz, amp_pa):
    """Random environmental clicks/pops (e.g., debris, distant snaps)."""
    rng = np.random.default_rng()
    sig = np.zeros(n_samples)
    num_impulses = int(rate_hz * (n_samples / fs))
    indices = rng.integers(0, n_samples, num_impulses)
    
    imp_len = int(0.002 * fs)  # 2 ms decay
    t_imp = np.arange(imp_len) / fs
    imp_shape = np.exp(-t_imp * 1500) * np.sin(2 * np.pi * 2500 * t_imp)
    
    for idx in indices:
        end_idx = min(idx + imp_len, n_samples)
        sig[idx:end_idx] += imp_shape[:end_idx-idx] * amp_pa
    return sig

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. MICROPHONE PHYSICS NOISE MODELS                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def apply_mic_resonance(signal_v, fs, freq_hz, q, amp):
    """Simulate diaphragm mechanical ringing using a damped harmonic oscillator."""
    duration = 0.015  # 15 ms ring
    n_samp = int(duration * fs)
    t = np.arange(n_samp) / fs
    omega = 2 * np.pi * freq_hz
    damping = omega / (2 * q)
    ring = np.exp(-damping * t) * np.sin(omega * t)
    
    ringed = fftconvolve(signal_v, ring, mode='full')[:len(signal_v)]
    return signal_v + amp * ringed

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. PREAMP CIRCUIT NOISE & DISTORTION MODELS                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def generate_flicker_noise(n_samples, fs, corner_freq_hz, white_psd_v2_hz):
    """1/f flicker noise scaled to match white noise PSD at the corner frequency."""
    rng = np.random.default_rng()
    white = rng.standard_normal(n_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    freqs[0] = freqs[1]
    
    scale = np.sqrt(corner_freq_hz / freqs)
    X_flicker = X * scale
    flicker = np.fft.irfft(X_flicker, n=n_samples)
    
    f_min = fs / n_samples
    f_max = fs / 2
    K = white_psd_v2_hz * corner_freq_hz
    theoretical_rms = np.sqrt(K * np.log(f_max / f_min))
    
    current_rms = np.std(flicker)
    if current_rms > 0:
        flicker = flicker * (theoretical_rms / current_rms)
    return flicker

def apply_thd_tanh(signal_v, thd_percent):
    """Harmonic distortion via tanh soft-clipping waveshaper."""
    drive = 1.0 + (thd_percent / 100.0) * 5.0
    return np.tanh(drive * signal_v) / drive

def apply_crosstalk(signals_4ch, crosstalk_db):
    """Channel bleed via a 4x4 mixing matrix."""
    cross_lin = 10 ** (crosstalk_db / 20.0)
    matrix = np.full((4, 4), cross_lin)
    np.fill_diagonal(matrix, 1.0)
    
    # Normalize rows to conserve total power
    row_sums = np.sum(matrix, axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    
    return signals_4ch @ matrix.T

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. MAIN EXECUTION PIPELINE                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main():
    cfg = load_json("config_noise.json")
    
    # 1. Load clean signals & metadata
    fs, clean_pa = wavfile.read(cfg['input_wav'])
    clean_pa = clean_pa.astype(np.float32)
    orig_meta = load_json(cfg['input_meta'])
    n_samples, n_channels = clean_pa.shape
    
    print(f"Loaded clean WAV: {n_samples} samples, {n_channels} channels, fs={fs} Hz")
    
    # 2. Acoustic Environment (Reverb + Colored Noise + Impulses)
    env_cfg = cfg['acoustic_environment']
    print("Applying acoustic environment...")
    
    rir = make_statistical_rir(env_cfg['rt60_s'], fs, env_cfg['early_reflections_count'])
    wet_pa = np.zeros_like(clean_pa)
    for i in range(n_channels):
        wet_pa[:, i] = fftconvolve(clean_pa[:, i], rir, mode='full')[:n_samples]
    
    colored_noise = make_colored_noise(n_samples, fs, env_cfg['colored_noise_slope'], env_cfg['colored_noise_rms_pa'])
    impulsive_noise = make_impulsive_noise(n_samples, fs, env_cfg['impulsive_noise_rate_hz'], env_cfg['impulsive_noise_amp_pa'])
    
    acoustic_noisy_pa = wet_pa + colored_noise[:, np.newaxis] + impulsive_noise[:, np.newaxis]
    
    # 3. Convert Pascals to Volts (Mic Sensitivity)
    sensitivity = cfg['mic_sensitivity_mv_pa']
    print(f"Converting to Volts (Sensitivity: {sensitivity} mV/Pa)...")
    signal_v = acoustic_noisy_pa * (sensitivity / 1000.0)
    
    # 4. Microphone Physics (Thermal Noise + Resonance)
    mic_cfg = cfg['mic_physics']
    print("Applying mic physics...")
    
    p_ein_pa = 20e-6 * (10 ** (mic_cfg['ein_dba'] / 20.0))
    v_ein_rms = p_ein_pa * (sensitivity / 1000.0)
    rng = np.random.default_rng()
    thermal_noise = rng.standard_normal((n_samples, n_channels)) * v_ein_rms
    signal_v += thermal_noise
    
    for i in range(n_channels):
        signal_v[:, i] = apply_mic_resonance(signal_v[:, i], fs, mic_cfg['resonance_freq_hz'], mic_cfg['resonance_q'], mic_cfg['resonance_amplitude'])
        
    # 5. Preamp Circuit (White Noise + Flicker + THD + DC Offset)
    preamp_cfg = cfg['preamp_circuit']
    print("Applying preamp circuit...")
    
    bw = fs / 2.0
    v_white_rms = preamp_cfg['white_noise_density_nv_root_hz'] * 1e-9 * np.sqrt(bw)
    white_noise = rng.standard_normal((n_samples, n_channels)) * v_white_rms
    signal_v += white_noise
    
    white_psd = (preamp_cfg['white_noise_density_nv_root_hz'] * 1e-9) ** 2
    for i in range(n_channels):
        flicker = generate_flicker_noise(n_samples, fs, preamp_cfg['flicker_noise_corner_hz'], white_psd)
        signal_v[:, i] += flicker
        
    for i in range(n_channels):
        signal_v[:, i] = apply_thd_tanh(signal_v[:, i], preamp_cfg['thd_percent'])
        
    signal_v += preamp_cfg['dc_offset_mv'] / 1000.0
    
    # 6. System Level (Crosstalk)
    print("Applying system crosstalk...")
    signal_v = apply_crosstalk(signal_v, preamp_cfg['crosstalk_db'])
    
    # 7. Save Noisy WAV (32-bit float Volts)
    wav_path = cfg['output_wav']
    wavfile.write(wav_path, fs, signal_v.astype(np.float32))
    print(f"Saved noisy WAV to: {wav_path}")
    
    # 8. Save Noisy Metadata JSON
    meta_path = cfg['output_meta']
    noisy_meta = {
        "original_metadata": orig_meta,
        "noise_chain_config": cfg,
        "signal_chain": [
            "1. Acoustic Reverberation (Statistical RIR + Early Reflections)",
            "2. Environmental Colored Noise (1/f^alpha)",
            "3. Environmental Impulsive Noise",
            "4. Mic Sensitivity Conversion (Pa -> V)",
            "5. Mic Thermal Noise (EIN)",
            "6. Mic Diaphragm Resonance",
            "7. Preamp White Noise",
            "8. Preamp Flicker (1/f) Noise",
            "9. Harmonic Distortion (THD via tanh)",
            "10. DC Offset",
            "11. Channel Crosstalk"
        ],
        "output_units": "Volts (32-bit float, NO ADC QUANTIZATION)",
        "derived_noise_levels": {
            "mic_thermal_noise_v_rms": float(v_ein_rms),
            "preamp_white_noise_v_rms": float(v_white_rms)
        }
    }
    
    save_json(meta_path, noisy_meta)
    print(f"Saved noisy metadata to: {meta_path}")

if __name__ == "__main__":
    main()