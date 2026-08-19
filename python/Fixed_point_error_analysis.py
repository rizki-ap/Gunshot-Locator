import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PHYSICAL MODELS (Floating-Point Baseline)
# ==========================================
def nwave_float(t, t_arrive, b, M, c, L, dP_ref, b_ref):
    beta = np.sqrt(M**2 - 1.0)
    T_N = (2.0 / (M * c)) * np.sqrt(b * L / beta)
    dP = dP_ref * np.sqrt(b_ref / b)
    p = np.zeros_like(t)
    mask = (t >= t_arrive) & (t < t_arrive + T_N)
    if mask.any():
        tau = t[mask] - t_arrive
        p[mask] = dP * (1.0 - 2.0 * tau / T_N)
    return p, T_N, dP

def friedlander_float(t, t_arrive, r, t_pos_ref, P_ref, R_ref):
    dP = P_ref * (R_ref / r)
    t_pos = t_pos_ref * (r / R_ref)**(1.0/3.0)
    p = np.zeros_like(t)
    mask = t >= t_arrive
    if mask.any():
        tau = t[mask] - t_arrive
        p[mask] = dP * (1.0 - tau/t_pos) * np.exp(-tau/t_pos)
    return p, t_pos, dP

# ==========================================
# 2. FIXED-POINT SIMULATION ENGINE
# ==========================================
class FixedPointSim:
    def __init__(self, frac_bits=16, total_bits=32):
        self.frac_bits = frac_bits
        self.total_bits = total_bits
        self.scale = 2**frac_bits
        self.max_val = (2**(total_bits-1) - 1) / self.scale
        self.min_val = -(2**(total_bits-1)) / self.scale
        
    def quantize(self, x):
        x_scaled = np.round(x * self.scale) / self.scale
        return np.clip(x_scaled, self.min_val, self.max_val)

    def nwave_fixed(self, t, t_arrive, b, M, c, L, dP_ref, b_ref):
        beta = np.sqrt(M**2 - 1.0) 
        T_N = self.quantize((2.0 / (M * c)) * np.sqrt(b * L / beta))
        dP = self.quantize(dP_ref * np.sqrt(b_ref / b))
        p = np.zeros_like(t)
        mask = (t >= t_arrive) & (t < t_arrive + T_N)
        if mask.any():
            tau = self.quantize(t[mask] - t_arrive)
            two_tau = self.quantize(2.0 * tau)
            ratio = self.quantize(two_tau / T_N)       
            term = self.quantize(1.0 - ratio)          
            p[mask] = self.quantize(dP * term)         
        return p

    def friedlander_fixed(self, t, t_arrive, r, t_pos_ref, P_ref, R_ref):
        dP = self.quantize(P_ref * (R_ref / r))
        t_pos = self.quantize(t_pos_ref * (r / R_ref)**(1.0/3.0))
        p = np.zeros_like(t)
        mask = t >= t_arrive
        if mask.any():
            tau = self.quantize(t[mask] - t_arrive)
            ratio = self.quantize(tau / t_pos)         
            exp_val = self.quantize(np.exp(-ratio))    
            one_minus = self.quantize(1.0 - ratio)
            term2 = self.quantize(one_minus * exp_val)
            p[mask] = self.quantize(dP * term2)
        return p

# ==========================================
# 3. METRICS
# ==========================================
def find_onset_time(t, p, threshold_ratio=0.1):
    peak = np.max(np.abs(p))
    if peak == 0: return np.nan
    thresh = peak * threshold_ratio
    crossings = np.where(np.abs(p) >= thresh)[0]
    if len(crossings) > 0:
        idx = crossings[0]
        if idx > 0:
            y1, y2 = np.abs(p[idx-1]), np.abs(p[idx])
            t1, t2 = t[idx-1], t[idx]
            return t1 + (thresh - y1) * (t2 - t1) / (y2 - y1 + 1e-12)
        return t[idx]
    return np.nan

def calculate_metrics(t, p_float, p_fixed):
    noise = p_float - p_fixed
    rmse = np.sqrt(np.mean(noise**2))
    sig_power = np.mean(p_float**2)
    noise_power = np.mean(noise**2)
    sqnr = 10 * np.log10(sig_power / (noise_power + 1e-20))
    peak_err = abs(np.max(p_float) - np.max(p_fixed))
    t_onset_float = find_onset_time(t, p_float, 0.1)
    t_onset_fixed = find_onset_time(t, p_fixed, 0.1)
    jitter_us = abs(t_onset_float - t_onset_fixed) * 1e6 
    return rmse, sqnr, peak_err, jitter_us

# ==========================================
# 4. EXECUTION
# ==========================================
FS = 100_000  
t = np.arange(0, 0.050, 1.0/FS) 
M, c, L = 2.5, 343.0, 0.023
b_sw, r_mb = 10.0, 200.0
dP0_sw, b0_sw = 7.5, 50.0
P_ref_mb, R_ref_mb, t_pos_ref = 200.0, 10.0, 0.003
t_arr_sw, t_arr_mb = 0.005, 0.015 

p_sw_float, _, _ = nwave_float(t, t_arr_sw, b_sw, M, c, L, dP0_sw, b0_sw)
p_mb_float, _, _ = friedlander_float(t, t_arr_mb, r_mb, t_pos_ref, P_ref_mb, R_ref_mb)

frac_bits_list = [8, 12, 16, 20, 24, 31] 
results = []

print(f"{'Format':<10} | {'Signal':<10} | {'RMSE (Pa)':<12} | {'SQNR (dB)':<12} | {'Peak Err':<12} | {'Timing Jitter':<15}")
print("-" * 85)

for fb in frac_bits_list:
    sim = FixedPointSim(frac_bits=fb, total_bits=32)
    p_sw_fix = sim.nwave_fixed(t, t_arr_sw, b_sw, M, c, L, dP0_sw, b0_sw)
    rmse_sw, sqnr_sw, pk_sw, jit_sw = calculate_metrics(t, p_sw_float, p_sw_fix)
    results.append([f"Q{31-fb}.{fb}", "Shockwave", rmse_sw, sqnr_sw, pk_sw, jit_sw])
    
    p_mb_fix = sim.friedlander_fixed(t, t_arr_mb, r_mb, t_pos_ref, P_ref_mb, R_ref_mb)
    rmse_mb, sqnr_mb, pk_mb, jit_mb = calculate_metrics(t, p_mb_float, p_mb_fix)
    results.append([f"Q{31-fb}.{fb}", "Muzzle Blast", rmse_mb, sqnr_mb, pk_mb, jit_mb])

df = pd.DataFrame(results, columns=['Format', 'Signal', 'RMSE', 'SQNR', 'Peak Err', 'Jitter (µs)'])
print(df.to_string(index=False))
