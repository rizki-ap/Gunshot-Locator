import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. HARDWARE PARAMETERS & FIXED-POINT SIM
# ==========================================
TOTAL_BITS = 32       
FRAC_BITS  = 20       
SCALE      = 2**FRAC_BITS
MAX_VAL    = (2**(TOTAL_BITS-1) - 1) / SCALE
MIN_VAL    = -(2**(TOTAL_BITS-1)) / SCALE

X_MIN, X_MAX = -8.0, 0.0 
NUM_SEGMENTS = 2048     
SEG_WIDTH    = (X_MAX - X_MIN) / NUM_SEGMENTS 

def quantize(x):
    x_scaled = np.round(x * SCALE) / SCALE
    return np.clip(x_scaled, MIN_VAL, MAX_VAL)

# ==========================================
# 2. LUT GENERATION
# ==========================================
lut_indices = np.arange(NUM_SEGMENTS)
lut_x0 = X_MIN + lut_indices * SEG_WIDTH
LUT_DATA = np.exp(lut_x0) 

print(f"Hardware Configuration:")
print(f"  Input Range      : [{X_MIN}, {X_MAX}]")
print(f"  LUT Depth        : {NUM_SEGMENTS} entries (Requires 1x 72Kb BRAM)")
print(f"  Segment Width    : {SEG_WIDTH:.6f}")

# ==========================================
# 3. THE "RTL" HARDWARE MODEL IN PYTHON
# ==========================================
def exp_hardware_model(x_float):
    x = np.atleast_1d(x_float).astype(float)
    
    # STAGE 1: Clamp
    x_clamped = np.clip(x, X_MIN, X_MAX)
    
    # STAGE 2: Index Extraction
    x_pos = x_clamped - X_MIN 
    idx = np.floor(x_pos / SEG_WIDTH).astype(int)
    idx = np.clip(idx, 0, NUM_SEGMENTS - 1) 
    
    # STAGE 3: LUT Lookup
    y_coarse = LUT_DATA[idx]
    
    # STAGE 4: Fine Residual Calculation
    x_base = idx * SEG_WIDTH 
    delta = quantize(x_pos - x_base) 
    
    # STAGE 5: Taylor Expansion (e^delta ≈ 1 + delta + delta^2/2)
    delta_sq_half = quantize((delta * delta) / 2.0)
    taylor_factor = quantize(1.0 + delta + delta_sq_half)
    
    # STAGE 6: Final Multiply
    y_final = quantize(y_coarse * taylor_factor)
    y_final[x < X_MIN] = 0.0 
    
    return y_final

# ==========================================
# 4. VERIFICATION & ERROR ANALYSIS
# ==========================================
x_test = np.linspace(-8.5, 0.5, 10000) 
y_true = np.exp(x_test)
y_hw   = exp_hardware_model(x_test)

abs_error = np.abs(y_true - y_hw)
max_error = np.max(abs_error)
rms_error = np.sqrt(np.mean(abs_error**2))
effective_bits = -np.log2(max_error + 1e-20)

print("\n" + "="*50)
print("PRECISION VERIFICATION RESULTS:")
print("="*50)
print(f"  Maximum Absolute Error : {max_error:.2e}")
print(f"  RMS Error              : {rms_error:.2e}")
print(f"  Effective Precision    : {effective_bits:.1f} bits")
print(f"  Target Precision       : 24.0 bits (Error < 5.96e-08)")
print(f"  STATUS                 : {'✅ PASS' if max_error < 5.96e-08 else '❌ FAIL'}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FPGA Hardware Model: Coarse-Fine exp() Function (2048-entry LUT + Taylor)', 
             fontsize=15, fontweight='bold')

axs[0, 0].plot(x_test, y_true, 'k-', lw=2, label='True $e^x$ (Float64)')
axs[0, 0].plot(x_test, y_hw, 'r--', lw=1, label='FPGA Model (Q11.20)')
axs[0, 0].set_title('Exponential Waveform Comparison')
axs[0, 0].set_xlabel('Input x'); axs[0, 0].set_ylabel('Output y')
axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(x_test, abs_error, 'b-', lw=1)
axs[0, 1].axhline(2**-24, color='red', ls='--', lw=2, label=f'24-bit limit ($2^{{-24}}$)')
axs[0, 1].set_title('Absolute Quantization & Approximation Error')
axs[0, 1].set_xlabel('Input x'); axs[0, 1].set_ylabel('Absolute Error')
axs[0, 1].set_yscale('log')
axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.3, which='both')

x_zoom = np.linspace(-0.01, 0, 500)
delta_zoom = x_zoom - (-0.01) 
taylor_1 = 1 + delta_zoom
taylor_2 = 1 + delta_zoom + delta_zoom**2 / 2
true_zoom = np.exp(delta_zoom)

axs[1, 0].plot(delta_zoom, true_zoom, 'k-', lw=2, label='True $e^\delta$')
axs[1, 0].plot(delta_zoom, taylor_1, 'g:', lw=2, label='1st Order (1 + $\delta$)')
axs[1, 0].plot(delta_zoom, taylor_2, 'r--', lw=2, label='2nd Order (1 + $\delta$ + $\delta^2/2$)')
axs[1, 0].set_title('Zoom: Taylor Expansion Accuracy for Fine Residual')
axs[1, 0].set_xlabel('Residual $\delta$'); axs[1, 0].set_ylabel('Value')
axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].axis('off')
resource_text = (
    "FPGA RESOURCE ESTIMATION (e.g., Xilinx Zynq 7020):\n\n"
    "1. LUT / BRAM:\n"
    "   - 2048 entries × 32 bits = 65,536 bits\n"
    "   - Maps to exactly ONE 72Kb Block RAM (BRAM36)\n\n"
    "2. DSP48 Slices (Math):\n"
    "   - delta * delta (1 DSP)\n"
    "   - y_coarse * taylor (1 DSP)\n"
    "   - Total: 2 DSP48E1 slices\n\n"
    "3. Throughput:\n"
    "   - Fully pipelined. 1 new result per clock cycle.\n"
    "   - Latency: ~4-6 clock cycles."
)
axs[1, 1].text(0.05, 0.5, resource_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axs[1, 1].set_title('FPGA Hardware Resource Footprint', fontweight='bold')

plt.tight_layout()
plt.show()

