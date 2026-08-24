import numpy as np
import json
import os
from scipy.io import wavfile

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION & PHYSICAL CONSTANTS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
C = 343.0  # Speed of sound (m/s)

def load_config(filepath="config_gunshot_generate_noiseless.txt"):
    with open(filepath, 'r') as f:
        return json.load(f)

BULLET_LIBRARY = {
    '7.62_NATO': dict(L=0.028, dP0_sw=7.5, b0_sw=50.0, P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003),
    '5.56_NATO': dict(L=0.023, dP0_sw=7.5, b0_sw=50.0, P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003/4.0),
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. ARRAY GEOMETRY & SCENE SETUP                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def make_tetrahedron(edge_length):
    raw = np.array([[0.000, 0.000, 1.000], [0.000, 0.943, -0.333],
                    [-0.816, -0.471, -0.333], [0.816, -0.471, -0.333]], dtype=float)
    scale = edge_length / np.linalg.norm(raw[0] - raw[1])
    return raw * scale

def shockwave_arrival(sensor_pos, bullet_origin, v_hat, M, c):
    beta = np.sqrt(M**2 - 1.0)
    r = sensor_pos - bullet_origin
    a = float(np.dot(r, v_hat))
    b = float(np.linalg.norm(r - a * v_hat))
    if a - b / beta < 0:
        raise ValueError(f"Sensor outside Mach cone (a={a:.1f} < b/beta={b/beta:.1f}).")
    t_arrive = (a + b * beta) / (M * c)
    t_emit = (a - b / beta) / (M * c)
    return t_arrive, t_emit, a, b

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. NOISELESS WAVEFORM GENERATION (PHYSICS)                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def nwave_waveform(t, t_arrive, b, M, c, L, dP0_sw, b0_sw):
    """Deterministic Whitham N-wave (Shockwave)."""
    beta = np.sqrt(M**2 - 1.0)
    T_N = (2.0 / (M * c)) * np.sqrt(b * L / beta)
    dP = dP0_sw * np.sqrt(b0_sw / b)
    p = np.zeros(len(t))
    mask = (t >= t_arrive) & (t < t_arrive + T_N)
    if mask.any():
        tau = t[mask] - t_arrive
        p[mask] = dP * (1.0 - 2.0 * tau / T_N)
    return p, T_N, dP

def friedlander_waveform(t, t_arrive, r, P_REF_MB, R_REF_MB, T_POS_REF):
    """Deterministic Friedlander blast (Muzzle Blast)."""
    dP = P_REF_MB * (R_REF_MB / r)
    t_pos = T_POS_REF * (r / R_REF_MB) ** (1.0/3.0)
    p = np.zeros(len(t))
    mask = t >= t_arrive
    if mask.any():
        tau = t[mask] - t_arrive
        p[mask] = dP * (1.0 - tau / t_pos) * np.exp(-tau / t_pos)
    return p, t_pos, dP

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. MAIN EXECUTION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main():
    cfg = load_config("config.txt")
    
    # Extract parameters
    L_ARRAY = cfg['L_ARRAY']
    FS = cfg['FS']
    CALIBER = cfg['CALIBER']
    M = cfg['MACH']
    RANGE = cfg['RANGE']
    Y_MISS = cfg['Y_MISS']
    
    bl = BULLET_LIBRARY[CALIBER]
    L_BULLET, dP0_sw, b0_sw = bl['L'], bl['dP0_sw'], bl['b0_sw']
    P_REF_MB, R_REF_MB, T_POS_REF = bl['P_REF_MB'], bl['R_REF_MB'], bl['T_POS_REF']

    # 1. Geometry
    mic_pos = make_tetrahedron(L_ARRAY)
    BULLET_ORIGIN = np.array([-RANGE, Y_MISS, 0.0])
    V_HAT = np.array([1.0, 0.0, 0.0])
    SHOOTER_POS = BULLET_ORIGIN.copy()

    # 2. Calculate Arrivals
    t_arr = np.zeros(4); t_emi = np.zeros(4); a_all = np.zeros(4); b_all = np.zeros(4)
    for i in range(4):
        t_arr[i], t_emi[i], a_all[i], b_all[i] = shockwave_arrival(mic_pos[i], BULLET_ORIGIN, V_HAT, M, C)
        
    r_mb_all = np.array([np.linalg.norm(p - SHOOTER_POS) for p in mic_pos])
    t_mb = r_mb_all / C

    # 3. Master Timeline
    PRE_ROLL = cfg.get('PRE_ROLL_S', 0.010)
    POST_ROLL = cfg.get('POST_ROLL_S', 0.150)
    t0 = min(t_arr.min(), t_mb.min()) - PRE_ROLL
    t1 = t_mb.max() + POST_ROLL
    t_master = np.arange(t0, t1, 1.0 / FS)

    # 4. Generate 4-Channel Noiseless Signals
    signals_4ch = np.zeros((len(t_master), 4), dtype=np.float32)
    metadata_channels = []

    print(f"Generating noiseless 4-channel WAV ({CALIBER}, Mach {M}, Range {RANGE}m)...")
    for i in range(4):
        # Shockwave
        sw_sig, T_N, dP_sw = nwave_waveform(t_master, t_arr[i], b_all[i], M, C, L_BULLET, dP0_sw, b0_sw)
        # Muzzle Blast
        mb_sig, t_pos, dP_mb = friedlander_waveform(t_master, t_mb[i], r_mb_all[i], P_REF_MB, R_REF_MB, T_POS_REF)
        
        # Superimpose
        combined = sw_sig + mb_sig
        signals_4ch[:, i] = combined
        
        # Store channel metadata
        metadata_channels.append({
            "mic_index": i,
            "position_m": mic_pos[i].tolist(),
            "shockwave_arrival_s": float(t_arr[i]),
            "shockwave_peak_Pa": float(dP_sw),
            "shockwave_duration_s": float(T_N),
            "muzzle_blast_arrival_s": float(t_mb[i]),
            "muzzle_blast_peak_Pa": float(dP_mb),
            "muzzle_blast_t_pos_s": float(t_pos),
            "range_to_shooter_m": float(r_mb_all[i])
        })

    # 5. Export WAV (32-bit float to preserve exact Pascal physics values)
    wav_path = cfg.get('OUTPUT_WAV', 'gunshot_simulation_4ch.wav')
    wavfile.write(wav_path, FS, signals_4ch)
    print(f"Saved 4-channel WAV to: {wav_path}")

    # 6. Export JSON Metadata
    meta_path = cfg.get('OUTPUT_META', 'gunshot_simulation_metadata.json')
    metadata = {
        "simulation_config": cfg,
        "physics_constants": {"speed_of_sound_m_s": C},
        "bullet_properties": bl,
        "array_geometry": {
            "edge_length_m": L_ARRAY,
            "mic_positions_m": mic_pos.tolist()
        },
        "scene_geometry": {
            "shooter_position_m": SHOOTER_POS.tolist(),
            "bullet_origin_m": BULLET_ORIGIN.tolist(),
            "trajectory_vector": V_HAT.tolist(),
            "range_m": RANGE,
            "miss_distance_m": Y_MISS,
            "mach_number": M
        },
        "audio_properties": {
            "sample_rate_hz": FS,
            "bit_depth": "32-bit float",
            "num_channels": 4,
            "duration_s": float(t_master[-1] - t_master[0]),
            "num_samples": len(t_master),
            "global_peak_Pa": float(np.max(np.abs(signals_4ch)))
        },
        "channel_details": metadata_channels
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {meta_path}")

if __name__ == "__main__":
    main()