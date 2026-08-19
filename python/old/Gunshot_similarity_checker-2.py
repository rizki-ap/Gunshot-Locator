import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================
# 1. LOAD OR GENERATE SIGNALS
# ==========================================
SR = 22050 
duration = 1.5 
t = np.linspace(0, duration, int(SR * duration), endpoint=False)

# Dummy Signals (Replace with librosa.load('your_file.wav', sr=SR) for real data)
np.random.seed(42)
noise_burst = np.exp(-t * 80) * np.random.randn(len(t)) 
resonance = np.exp(-t * 25) * np.sin(2 * np.pi * 120 * t)
y_real = (noise_burst * 0.6 + resonance * 0.8).astype(np.float32)

t_shifted = t - 0.005 
noise_burst_syn = np.exp(-t * 60) * np.random.randn(len(t)) * 0.9
resonance_syn = np.exp(-t * 30) * np.sin(2 * np.pi * 125 * t) * 1.1 
y_synth = (noise_burst_syn * 0.6 + resonance_syn * 0.8).astype(np.float32)
y_synth = y_synth * 0.85 + 0.02 

# Preprocessing
y_real = y_real / np.max(np.abs(y_real))
y_synth = y_synth / np.max(np.abs(y_synth))
max_len = max(len(y_real), len(y_synth))
y_real_padded = np.pad(y_real, (0, max_len - len(y_real)), mode='constant')
y_synth_padded = np.pad(y_synth, (0, max_len - len(y_synth)), mode='constant')

# ==========================================
# 2. CALCULATE SIMILARITY METRICS
# ==========================================
results = {}
eps = 1e-10

# Time-Domain
rmse = np.sqrt(np.mean((y_real_padded - y_synth_padded) ** 2))
results['RMSE (Time)'] = rmse

xcorr = signal.correlate(y_real_padded, y_synth_padded, mode='full')
norm_factor = np.linalg.norm(y_real_padded) * np.linalg.norm(y_synth_padded)
xcorr_norm = xcorr / norm_factor
results['Cross-Correlation'] = np.max(xcorr_norm)

# Frequency-Domain
stft_real = np.abs(librosa.stft(y_real_padded, n_fft=2048, hop_length=512))
stft_synth = np.abs(librosa.stft(y_synth_padded, n_fft=2048, hop_length=512))
lsd = np.sqrt(np.mean((np.log10(stft_real + eps) - np.log10(stft_synth + eps)) ** 2))
results['Log-Spectral Dist'] = lsd

mel_real_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y_real_padded, sr=SR), ref=np.max)
mel_synth_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y_synth_padded, sr=SR), ref=np.max)
results['Mel-Spec MSE'] = np.mean((mel_real_db - mel_synth_db) ** 2)

mel_real_norm = (mel_real_db - mel_real_db.min()) / (mel_real_db.max() - mel_real_db.min() + eps)
mel_synth_norm = (mel_synth_db - mel_synth_db.min()) / (mel_synth_db.max() - mel_synth_db.min() + eps)
min_frames = min(mel_real_norm.shape[1], mel_synth_norm.shape[1])
results['Mel-Spec SSIM'] = ssim(mel_real_norm[:, :min_frames], mel_synth_norm[:, :min_frames], data_range=1.0)

# Perceptual
mfcc_real_mean = np.mean(librosa.feature.mfcc(y=y_real_padded, sr=SR, n_mfcc=13), axis=1)
mfcc_synth_mean = np.mean(librosa.feature.mfcc(y=y_synth_padded, sr=SR, n_mfcc=13), axis=1)
results['MFCC Cosine Sim'] = 1 - cosine(mfcc_real_mean, mfcc_synth_mean)

dtw_distance, _ = fastdtw(mfcc_real_mean.reshape(-1,1), mfcc_synth_mean.reshape(-1,1), dist=1)
results['DTW Distance'] = dtw_distance

# ==========================================
# 3. AUTOMATIC PASS/FAIL EVALUATION
# ==========================================
PASS_FAIL_THRESHOLDS = {
    'MFCC Cosine Sim': {'min': 0.85, 'target': 0.95},
    'Mel-Spec SSIM':   {'min': 0.60, 'target': 0.75},
    'Log-Spectral Dist': {'max': 3.0, 'target': 1.5},
    'Cross-Correlation': {'min': 0.70, 'target': 0.85},
    'RMSE (Time)':     {'max': 0.10, 'target': 0.05}
}

print("\n" + "="*80)
print("GUNSHOT SYNTHESIS QUALITY REPORT")
print("="*80)
print(f"{'Metric':<20} | {'Value':<12} | {'Status':<10} | {'Target'}")
print("-"*80)

for metric_name, thresholds in PASS_FAIL_THRESHOLDS.items():
    value = results.get(metric_name, 0)
    
    # Determine Status
    if 'min' in thresholds:
        status = "✅ PASS" if value >= thresholds['min'] else "❌ FAIL"
        target_str = f"> {thresholds['target']}"
    elif 'max' in thresholds:
        status = "✅ PASS" if value <= thresholds['max'] else "❌ FAIL"
        target_str = f"< {thresholds['target']}"
    else:
        status = "⚠️ CHECK"
        target_str = "N/A"

    print(f"{metric_name:<20} | {value:<12.4f} | {status:<10} | {target_str}")

print("="*80)
overall_status = "EXCELLENT" if all(
    (results[m] >= t['min'] if 'min' in t else results[m] <= t['max'])
    for m, t in PASS_FAIL_THRESHOLDS.items()
) else "NEEDS IMPROVEMENT"
print(f"OVERALL VERDICT: {overall_status}")
print("="*80)
