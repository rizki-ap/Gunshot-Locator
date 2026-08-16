# ╔═════════════════════════════════════════════╗
# ║  SECTION 1.1 — Imports & Physical Constants ║
# ╚═════════════════════════════════════════════╝
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Physical constants ─────────────────────────────────────────────────────────
C     = 343.0      # Speed of sound (m/s) -- ISA sea level
P_ATM = 101325.0   # Atmospheric pressure (Pa)


# ╔══════════════════════════════════════════════════════╗
# ║  SECTION 1.2 — Tetrahedral Microphone Array Geometry ║
# ╚══════════════════════════════════════════════════════╝
#  Regular tetrahedron from alternating +-1 cube-vertex coordinates:
#
#     (1,1,1)  (1,-1,-1)  (-1,1,-1)  (-1,-1,1)
#
#  All 6 edges of this subset have identical length 2*sqrt(2).
#  Scale by  L / (2*sqrt(2))  to hit the target aperture.

L_ARRAY = 0.30        # Tetrahedral edge length (m) -- ASSUMED SENSOR GEOMETRY

def make_tetrahedron(edge_length):
    raw = np.array([[0.000, 0.000, 1.000], [0.000, 0.943, -0.333],
                     [-0.816, -0.471, -0.333], [0.816, -0.471, -0.333]], dtype=float)
    scale = edge_length / np.linalg.norm(raw[0] - raw[1])   # raw edge = 2*sqrt(2)
    return raw * scale

mic_pos = make_tetrahedron(L_ARRAY)
COLORS  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

print("Microphone positions (m):")
for i, p in enumerate(mic_pos):
    print(f"  M{i}: [{p[0]:+.5f}, {p[1]:+.5f}, {p[2]:+.5f}]")
print("\nEdge-length verification:")
for i in range(4):
    for j in range(i+1, 4):
        d = np.linalg.norm(mic_pos[i] - mic_pos[j])
        print(f"  M{i}-M{j}: {d:.6f} m  {'OK' if abs(d-L_ARRAY)<1e-3 else 'MISMATCH'}")

# 3D array plot
fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title('Tetrahedral Microphone Array (0.3 m edge)', fontweight='bold')
for i, (p, col) in enumerate(zip(mic_pos, COLORS)):
    ax.scatter(*p, color=col, s=180, zorder=10)
    ax.text(p[0]+0.02, p[1]+0.02, p[2]+0.02,
            f'M{i} ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})',
            fontsize=8, color=col)
for i in range(4):
    for j in range(i+1, 4):
        ax.plot3D(*zip(mic_pos[i], mic_pos[j]),
                  color='gray', lw=0.8, alpha=0.5, linestyle='--')
m = 0.23
ax.set_xlim([-m,m]); ax.set_ylim([-m,m]); ax.set_zlim([-m,m])
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
plt.tight_layout(); plt.show()


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 2.1 — Sample Rate, Bit Depth & Mic Frequency Response ║
# ╚════════════════════════════════════════════════════════════════╝
from scipy.signal import butter, sosfiltfilt, fftconvolve

FS         = 100_000   # Sample rate (Hz) -- ASSUMED ADC SETTING
BIT_DEPTH  = 16         # ADC quantization depth -- ASSUMED ADC SETTING
MIC_LO, MIC_HI = 40.0, 20000.0   # mic passband (Hz) -- ASSUMED SENSOR RESPONSE

def mic_bandpass(x, fs, lo=MIC_LO, hi=MIC_HI, order=4):
    """Simulate microphone/ADC anti-alias frequency response."""
    sos = butter(order, [lo, min(hi, fs/2*0.99)], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def quantize(x, bits=BIT_DEPTH, full_scale=1.0):
    """Simulate ADC quantization."""
    levels = 2 ** (bits - 1)
    return np.round(np.clip(x, -full_scale, full_scale) * levels) / levels

print(f"ADC settings: fs={FS/1000:.0f} kHz, {BIT_DEPTH}-bit, "
      f"mic passband [{MIC_LO:.0f}, {MIC_HI:.0f}] Hz")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2.2 — Reference Sensitivity Targets (measured from a real recording)║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#  These were MEASURED from a real recording (glock3_ch0.wav) -- see the
#  "Real vs Simulated" comparison later in this notebook. Re-derive them from
#  your own reference file if the mic/preamp/gain chain changes.

REAL_NOISE_RMS   = 0.000595   # measured background RMS (normalized units)
REAL_RT60        = 1.32       # measured reverb decay time (s)
REAL_PEAK_NORM   = 0.098      # measured peak amplitude (normalized units)
REAL_NOISE_SLOPE = -2.46      # measured PSD slope (brown-ish noise)

print("Reference sensitivity targets loaded from real-recording calibration.")


# ╔═════════════════════════════════════════╗
# ║  SECTION 3.1 — Bullet / Caliber Library ║
# ╚═════════════════════════════════════════╝
#  Friedlander & Whitham scaling constants are caliber-specific. The 5.56 entry
#  below is a rough fit derived from only 2 real Ruger recordings -- treat it
#  as a starting point, not a fully validated reference. Switch CALIBER to
#  '7.62_NATO' to fall back to the original long-range reference values.

BULLET_LIBRARY = {
    '7.62_NATO': dict(L=0.028, dP0_sw=7.5, b0_sw=50.0,
                       P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003),
    '5.56_NATO': dict(L=0.023, dP0_sw=7.5, b0_sw=50.0,   # dP0_sw not independently re-fit
                       P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003/4.0),
}
CALIBER = '5.56_NATO'   # Ruger .223/5.56 -- switch to '7.62_NATO' for the NATO reference
bl = BULLET_LIBRARY[CALIBER]

L_BULLET  = bl['L']           # Projectile length (m) -- scales N-wave duration
dP0_sw    = bl['dP0_sw']      # Reference shockwave peak overpressure (Pa)
b0_sw     = bl['b0_sw']       # Reference miss distance for dP0_sw (m)
P_REF_MB  = bl['P_REF_MB']    # Reference muzzle blast overpressure (Pa)
R_REF_MB  = bl['R_REF_MB']    # Reference range for P_REF_MB (m)
T_POS_REF = bl['T_POS_REF']   # Reference Friedlander positive-phase duration (s)

print(f"Bullet property: {CALIBER}")
print(f"  L_BULLET  = {L_BULLET*1000:.1f} mm")
print(f"  P_REF_MB  = {P_REF_MB:.0f} Pa @ {R_REF_MB:.0f} m, T_POS_REF = {T_POS_REF*1e3:.2f} ms")


# ╔═════════════════════════════════════════════════════╗
# ║  SECTION 3.2 — Assumed Bullet Velocity (Mach Number)║
# ╚═════════════════════════════════════════════════════╝
M        = 2.5      # Mach number -- ASSUMED BULLET VELOCITY
V_BULLET = M * C     # Bullet speed (m/s)

# Mach cone geometry
MU   = np.arcsin(1.0 / M)        # Mach half-angle  mu = arcsin(1/M)
BETA = np.sqrt(M**2 - 1.0)       # beta = sqrt(M^2-1)

print(f"Bullet         : Mach {M} = {V_BULLET:.0f} m/s")
print(f"Mach angle mu  : {np.degrees(MU):.2f} deg -> Mach cone angle")
print(f"beta = sqrt(M^2-1): {BETA:.4f}  -> Mach wave parameter, tangent of Mach angle")
print(f"Array aperture : {L_ARRAY} m  ->  max direct TDOA = {L_ARRAY/C*1e6:.1f} us")


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4.1 — Shooter Origin, Trajectory Direction & Miss Distance║
# ╚════════════════════════════════════════════════════════════════════╝
#  Shooter at (-RANGE, Y_MISS, 0).  Array centroid at origin.
#  Bullet flies in +x direction with Y_MISS lateral miss distance.
#
#   Y
#   |        * shooter (-RANGE, Y_MISS)
#   |          -----------------------------> bullet velocity
#   |
#   0------------------------------------ X
#  array (0,0,0)

RANGE  = 200.0     # Along-track shooter range (m)      -- ASSUMED TRAJECTORY
Y_MISS = 10.0      # Bullet clears array by this much (m) -- ASSUMED TRAJECTORY

BULLET_ORIGIN = np.array([-RANGE, Y_MISS, 0.0])   # bullet position at t=0 (also muzzle position)
V_HAT         = np.array([ 1.0,   0.0,   0.0])    # bullet velocity unit vector

# The muzzle blast originates from the same point the bullet trajectory starts:
# the shooter IS the bullet's origin.
SHOOTER_POS = BULLET_ORIGIN.copy()

print(f"Shooter / bullet origin : [{SHOOTER_POS[0]:.1f}, {SHOOTER_POS[1]:.1f}, {SHOOTER_POS[2]:.1f}] m")
print(f"Trajectory direction    : {V_HAT}")
print(f"Range (along-track)     : {RANGE:.1f} m   |   Miss distance: {Y_MISS:.1f} m")


# ╔══════════════════════════════════════════╗
# ║  SECTION 5.1 — Shockwave Arrival Physics ║
# ╚══════════════════════════════════════════╝
#  DERIVATION
#  ----------
#  Bullet position at emission time t_e:  p(t_e) = p0 + V_BULLET . t_e . v_hat
#
#  Mach cone condition -- the shockwave envelope satisfies:
#      cos(theta) = (s - p(t_e)) . v_hat / |s - p(t_e)| = 1/M
#
#  For sensor s and bullet origin p0:
#      r = s - p0
#      a = r . v_hat                (along-track distance)
#      b = |r - a.v_hat|            (perp. distance to trajectory)
#
#  Solving the Mach cone condition:
#      t_emit   = (a - b/beta) / (M.c)
#      t_arrive = (a + b.beta) / (M.c)     beta = sqrt(M^2-1)

def shockwave_arrival(sensor_pos, bullet_origin, v_hat, M, c):
    """Returns (t_arrive, t_emit, a, b) for a sensor inside the Mach cone."""
    beta = np.sqrt(M**2 - 1.0)
    r    = sensor_pos - bullet_origin
    a    = float(np.dot(r, v_hat))
    b    = float(np.linalg.norm(r - a * v_hat))
    if a - b / beta < 0:
        raise ValueError(
            f"Sensor outside Mach cone (a={a:.1f} < b/beta={b/beta:.1f}). "
            "Increase RANGE or reduce Y_MISS."
        )
    t_arrive = (a + b * beta) / (M * c)
    t_emit   = (a - b / beta) / (M * c)
    return t_arrive, t_emit, a, b

# Per-mic arrivals
t_arr = np.zeros(4);  t_emi = np.zeros(4)
a_all = np.zeros(4);  b_all = np.zeros(4)

print("Shockwave arrival analysis")
print("-" * 74)
print(f"{'Mic':<5} {'b (m)':>10} {'a (m)':>12} {'t_emit (ms)':>14} {'t_arrive (ms)':>15}")
for i, p in enumerate(mic_pos):
    ta, te, a, b = shockwave_arrival(p, BULLET_ORIGIN, V_HAT, M, C)
    t_arr[i]=ta; t_emi[i]=te; a_all[i]=a; b_all[i]=b
    print(f"  M{i}   {b:>10.5f}  {a:>12.4f}  {te*1e3:>14.6f}  {ta*1e3:>15.6f}")

t_ref, _, a_ref, b_ref = shockwave_arrival(np.zeros(3), BULLET_ORIGIN, V_HAT, M, C)
print(f"\n  Centroid:  b={b_ref:.4f} m  |  t_ref={t_ref*1e3:.6f} ms")

bullet_x_at_emit = -RANGE + V_BULLET * t_emi.mean()
bullet_x_at_arrv = -RANGE + V_BULLET * t_arr.mean()
print(f"\n  Bullet x at emission  : {bullet_x_at_emit:.1f} m  (relative to array)")
print(f"  Bullet x at detection : {bullet_x_at_arrv:.1f} m  (bullet already past array)")

print(f"\nTrue TDOAs (relative to M0):")
print(f"  {'Pair':<8} {'tau (us)':>12}")
for i in range(1,4):
    print(f"  (0,{i})    {(t_arr[i]-t_arr[0])*1e6:>+12.3f}")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.2 — Scene Geometry Plot                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# -- Left: full scene, top view (XY) --------------------------------------------
ax = axes[0]
x_traj   = np.linspace(BULLET_ORIGIN[0], BULLET_ORIGIN[0]+1400, 600)
cone_half = np.abs(x_traj - BULLET_ORIGIN[0]) * np.tan(MU)

ax.fill_between(x_traj, Y_MISS-cone_half, Y_MISS+cone_half,
                alpha=0.12, color='darkorange')
ax.plot(x_traj, Y_MISS+cone_half, '--', color='darkorange', lw=1.0, alpha=0.7,
        label='Mach cone')
ax.plot(x_traj, Y_MISS-cone_half, '--', color='darkorange', lw=1.0, alpha=0.7)

ax.annotate('', xy=(250, Y_MISS), xytext=BULLET_ORIGIN[:2],
            arrowprops=dict(arrowstyle='->', color='crimson', lw=2.2))
ax.text(-700, Y_MISS+4, f'Mach {M}  ->', color='crimson',
        fontsize=10, fontweight='bold')
ax.scatter(*BULLET_ORIGIN[:2], color='darkred', s=250, zorder=8,
           marker='*', label=f'Shooter ({RANGE:.0f} m)')
ax.scatter([0], [0], color='black', s=120, zorder=8, marker='^',
           label='Array centroid')

ax.annotate('', xy=(0, Y_MISS), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(8, Y_MISS/2, f'{Y_MISS:.0f} m\nmiss', color='gray',
        fontsize=9, va='center')
ax.plot([0,0], [0, Y_MISS], ':', color='gray', lw=1.0)

emission_x = -RANGE + V_BULLET * t_emi.mean()
r_wave     = b_all.mean() * M / BETA
theta_arc  = np.linspace(-np.pi*0.55, np.pi*0.55, 80)
ax.plot(emission_x + r_wave*np.cos(theta_arc),
        Y_MISS    + r_wave*np.sin(theta_arc),
        color='steelblue', lw=1.6, alpha=0.8, label='Wavefront arc')

for p, col in zip(mic_pos, COLORS):
    ax.scatter(p[0], p[1], color=col, s=70, zorder=9)

ax.set_xlim([-1060, 300]);  ax.set_ylim([-10, 130])
ax.set_xlabel('X (m)', fontsize=11); ax.set_ylabel('Y (m)', fontsize=11)
ax.set_title(f'Scene Overview -- Top View\nMach {M} bullet, {RANGE:.0f} m range, {Y_MISS:.0f} m miss',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3);  ax.set_aspect('equal')

# -- Right: array close-up with wavefront direction ------------------------------
ax2 = axes[1]
m   = 0.22
for i, (p, col) in enumerate(zip(mic_pos, COLORS)):
    ax2.scatter(p[0], p[1], color=col, s=220, zorder=9)
    ox = 0.012 if p[0] >= 0 else -0.075
    ax2.annotate(f'M{i}\n({p[0]:+.3f},{p[1]:+.3f})',
                 (p[0]+ox, p[1]+0.008), fontsize=8, color=col, fontweight='bold')
for i in range(4):
    for j in range(i+1, 4):
        ax2.plot([mic_pos[i,0], mic_pos[j,0]],
                 [mic_pos[i,1], mic_pos[j,1]],
                 'gray', lw=0.8, alpha=0.4, ls='--')

dx = 0 - emission_x;  dy = 0 - Y_MISS
nn = np.sqrt(dx**2+dy**2);  dx /= nn;  dy /= nn
ax2.annotate('', xy=(0.13*dx, 0.13*dy), xytext=(-0.13*dx, -0.13*dy),
             arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.2))
ang = np.degrees(np.arctan2(dy, dx))
ax2.text(-0.20, 0.10, f'Wavefront\n({ang:.1f} deg)',
         color='darkorange', fontsize=8)
ax2.scatter(0, 0, color='black', s=60, marker='+', zorder=8)
ax2.set_xlim([-m,m]);  ax2.set_ylim([-m,m])
ax2.set_xlabel('X (m)', fontsize=11); ax2.set_ylabel('Y (m)', fontsize=11)
ax2.set_title('Array Close-up (XY projection)',
              fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3);  ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('scene_geometry.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.3 — N-wave (Shockwave) Signal Generation (Whitham weak-shock model)  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  The N-wave is the far-field acoustic signature of a supersonic projectile:
#      p(tau) = dP * (1 - 2*tau/T_N)   for 0 <= tau <= T_N     tau = t - t_arrive
#      p(tau) = 0                        elsewhere
#
#  Whitham far-field scaling (Whitham 1952, Comm. Pure Appl. Math. 5:301):
#      Duration:       T_N = (2 / M.c) * sqrt(b . L_bullet / beta)
#      Peak pressure:  dP  ~ 1/sqrt(b)  (cylindrical-wave spreading)
#      Reference:      dP ~ 7.5 Pa at 50 m for 7.62x51 NATO, Mach 3
#                      (consistent with Bass et al. 1995, Stoughton 1997)

def nwave_duration(b, M, c, L):
    """Whitham far-field N-wave duration (s)."""
    return (2.0 / (M * c)) * np.sqrt(b * L / np.sqrt(M**2 - 1.0))

def nwave_peak_pressure(b, b_ref=b0_sw, dP_ref=dP0_sw):
    """Peak overpressure (Pa) via 1/sqrt(b) cylindrical spreading."""
    return dP_ref * np.sqrt(b_ref / b)

def generate_nwave(t, t_arrive, b, M, c, L, snr_db=20.0, rng=None):
    """Generate sampled N-wave pressure signal. Returns (pressure[Pa], T_N[s], dP[Pa])."""
    if rng is None:
        rng = np.random.default_rng(0)
    T_N = nwave_duration(b, M, c, L)
    dP  = nwave_peak_pressure(b)
    p   = np.zeros(len(t))
    mask = (t >= t_arrive) & (t < t_arrive + T_N)
    if mask.any():
        tau      = t[mask] - t_arrive
        p[mask]  = dP * (1.0 - 2.0 * tau / T_N)
    noise_amp = dP / (10.0 ** (snr_db / 20.0))
    p += noise_amp * rng.standard_normal(len(t))
    return p, T_N, dP

# Build time array centred on arrivals
t_start = t_arr.min() - 0.002     # 2 ms pre-roll
t_end   = t_arr.max() + 0.020     # 20 ms post-roll
t_sig   = np.arange(t_start, t_end, 1.0 / FS)

rng     = np.random.default_rng(seed=42)
signals = [];  T_Ns = [];  dPs = []

print(f"N-wave parameters (SNR = 20 dB, fs = {FS//1000} kHz)")
print(f"{'Mic':<5} {'b (m)':>10} {'T_N (ms)':>12} {'dP (Pa)':>10}")
for i in range(4):
    sig, T_N, dP = generate_nwave(
        t_sig, t_arr[i], b_all[i], M, C, L_BULLET, snr_db=20, rng=rng)
    signals.append(sig);  T_Ns.append(T_N);  dPs.append(dP)
    print(f"  M{i}   {b_all[i]:>10.4f}  {T_N*1e3:>12.4f}  {dP:>10.3f}")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.4 — N-wave Signal Plots                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
fig, axes_s = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

t_rel_ms = (t_sig - t_ref) * 1e3    # ms relative to centroid arrival

WIN_LO, WIN_HI = -2.0, 10.0
mask_w = (t_rel_ms >= WIN_LO) & (t_rel_ms <= WIN_HI)

for i in range(4):
    ax = axes_s[i]
    ax.plot(t_rel_ms[mask_w], signals[i][mask_w],
            color=COLORS[i], lw=1.6, label=f'M{i}')
    arr_rel = (t_arr[i] - t_ref) * 1e3
    ax.axvline(arr_rel, color=COLORS[i], ls='--', lw=1.4, alpha=0.9,
               label=f'Arrival: {arr_rel:+.4f} ms')
    ax.axhline(0,       color='black',    lw=0.5, alpha=0.35)
    ax.axhline( dPs[i], color=COLORS[i],  lw=0.6, ls=':', alpha=0.4)
    ax.axhline(-dPs[i], color=COLORS[i],  lw=0.6, ls=':', alpha=0.4)
    ax.set_ylabel('Pressure (Pa)', fontsize=10)
    ax.set_ylim([-dPs[i]*2.2, dPs[i]*2.2])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)

axes_s[-1].set_xlabel('Time relative to array centroid arrival (ms)', fontsize=11)
axes_s[0].set_title(
    f'N-wave Shockwave Signals -- Mach {M}, {RANGE:.0f} m range, '
    f'{Y_MISS:.0f} m miss distance\n'
    f'T_N ~ {T_Ns[0]*1e3:.2f} ms  |  dP ~ {dPs[0]:.1f} Pa  |  '
    f'SNR = 20 dB  |  fs = {FS//1000} kHz',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('nwave_signals.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.5 — Muzzle Blast Physics & Arrival Times                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  TWO ACOUSTIC EVENTS PER SHOT:
#
#  (1) Shockwave  -- Mach cone along bullet trajectory (arrives FIRST)
#       t_SW(s) = (a + b.beta) / (M.c)            [Section 5.1]
#
#  (2) Muzzle blast -- spherical wave from SHOOTER position at t=0 (arrives SECOND)
#       t_MB(s) = |s - s_shooter| / c
#
#  The shockwave always precedes the muzzle blast (since beta = sqrt(M^2-1) > 0):
#       Dt = t_MB - t_SW > 0  for all M > 1
#
#  --- What each event encodes -----------------------------------------------
#   SW  TDOAs  ->  bullet TRAJECTORY direction  (azimuth + elevation of v_hat)
#   MB  TDOAs  ->  shooter DIRECTION            (azimuth + elevation to source)
#   Dt         ->  shooter RANGE                (PILAR-V formula, Section 10)
#
#  SHOOTER_POS was already defined in Section 4.1 (= BULLET_ORIGIN).

r_mb_all  = np.array([np.linalg.norm(p - SHOOTER_POS) for p in mic_pos])
t_mb      = r_mb_all / C                                   # per-mic arrival (s)
t_mb_ref  = float(np.linalg.norm(SHOOTER_POS) / C)        # centroid arrival (s)

print("Muzzle blast arrivals")
print(f"  Shooter position : [{SHOOTER_POS[0]:.0f}, {SHOOTER_POS[1]:.0f}, "
      f"{SHOOTER_POS[2]:.0f}] m  "
      f"-> range = {np.linalg.norm(SHOOTER_POS):.2f} m")
print(f"  {'Mic':<5} {'Range (m)':>12} {'t_MB (ms)':>14}")
for i in range(4):
    print(f"  M{i}   {r_mb_all[i]:>12.4f}  {t_mb[i]*1e3:>14.4f}")
print(f"  Centroid  {np.linalg.norm(SHOOTER_POS):>12.4f}  {t_mb_ref*1e3:>14.4f}")

print(f"\n{'-'*58}")
print(f"  {'Pair':<8} {'tau_SW (us)':>14}   {'tau_MB (us)':>14}")
print(f"  {'':8} {'<- bullet dir':>14}   {'<- shooter dir':>14}")
for i in range(1, 4):
    sw_us = (t_arr[i] - t_arr[0]) * 1e6
    mb_us = (t_mb[i]  - t_mb[0])  * 1e6
    print(f"  (0,{i})    {sw_us:>+14.3f}   {mb_us:>+14.3f}")

Dt = t_mb_ref - t_ref    # timing differential (always > 0)
print(f"\n  t_SW  (centroid) : {t_ref*1e3:.3f} ms")
print(f"  t_MB  (centroid) : {t_mb_ref*1e3:.3f} ms")
print(f"  Dt = t_MB - t_SW : {Dt*1e3:.4f} ms  <- key input to PILAR-V range formula")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.6 — Friedlander (Muzzle Blast) Signal Generation                     ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Friedlander (1946) free-field blast waveform:
#      p(tau) = dP . (1 - tau/t_pos) . exp(-tau/t_pos)      tau >= 0
#      p(tau) = 0                                            tau < 0
#
#  Hopkinson-Cranz scaling (cube-root blast law):
#      dP(r)    = dP_ref  . (r_ref / r)            [spherical 1/r spreading]
#      t_pos(r) = t_pos_ref . (r / r_ref)^(1/3)    [blast cube-root law]
#
#  FIX vs. original notebook: P_REF_MB / R_REF_MB / T_POS_REF are NOT
#  redefined here -- they already come from the BULLET_LIBRARY entry chosen
#  in Section 3.1, so changing CALIBER there now actually changes the muzzle
#  blast waveform generated here (previously this cell silently overwrote
#  them with hardcoded 7.62 NATO values regardless of CALIBER).

def friedlander_params(r):
    """Peak overpressure (Pa) and positive phase duration (s) at range r."""
    dP    = P_REF_MB  * (R_REF_MB / r)
    t_pos = T_POS_REF * (r / R_REF_MB) ** (1.0/3.0)
    return dP, t_pos

def generate_friedlander(t, t_arrive, r, snr_db=20.0, rng=None):
    """Sampled Friedlander muzzle blast waveform. Returns (pressure[Pa], t_pos[s], dP[Pa])."""
    if rng is None:
        rng = np.random.default_rng(0)
    dP, t_pos = friedlander_params(r)
    p = np.zeros(len(t))
    mask = t >= t_arrive
    if mask.any():
        tau     = t[mask] - t_arrive
        p[mask] = dP * (1.0 - tau/t_pos) * np.exp(-tau/t_pos)
    noise_amp = dP / (10.0 ** (snr_db / 20.0))
    p += noise_amp * rng.standard_normal(len(t))
    return p, t_pos, dP

# Separate time array for the muzzle blast window (centred on MB arrivals)
PAD_MB_PRE  = 0.005   # 5 ms pre-roll
PAD_MB_POST = 0.120   # 120 ms post-roll (covers >= 8 x t_pos)

t_mb_sig = np.arange(t_mb.min() - PAD_MB_PRE,
                      t_mb.max() + PAD_MB_POST,
                      1.0 / FS)

rng_mb     = np.random.default_rng(seed=99)
signals_mb = [];  t_pos_all = [];  dP_mb_all = []

print(f"Friedlander parameters  "
      f"(r_ref={R_REF_MB:.0f} m, dP_ref={P_REF_MB:.0f} Pa, "
      f"t_pos_ref={T_POS_REF*1e3:.2f} ms)")
print(f"{'Mic':<5} {'Range (m)':>10} {'dP (Pa)':>10} {'t_pos (ms)':>12}")
for i in range(4):
    sig, tp, dP = generate_friedlander(
        t_mb_sig, t_mb[i], r_mb_all[i], snr_db=20, rng=rng_mb)
    signals_mb.append(sig)
    t_pos_all.append(tp)
    dP_mb_all.append(dP)
    print(f"  M{i}   {r_mb_all[i]:>10.3f}  {dP:>10.4f}  {tp*1e3:>12.3f}")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.7 — Muzzle Blast Signal Plots                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
fig, axes_mb = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

t_mb_rel_ms = (t_mb_sig - t_mb_ref) * 1e3
WIN_MB = (-5.0, 80.0)
mask_mb = (t_mb_rel_ms >= WIN_MB[0]) & (t_mb_rel_ms <= WIN_MB[1])

for i in range(4):
    ax = axes_mb[i]
    ax.plot(t_mb_rel_ms[mask_mb], signals_mb[i][mask_mb],
            color=COLORS[i], lw=1.5, label=f'M{i}')
    arr_rel = (t_mb[i] - t_mb_ref) * 1e3
    ax.axvline(arr_rel, color=COLORS[i], ls='--', lw=1.4, alpha=0.9,
               label=f'Arrival: {arr_rel:+.3f} ms')
    ax.axhline(0, color='black', lw=0.5, alpha=0.35)
    ax.axvline(arr_rel + t_pos_all[i]*1e3, color=COLORS[i],
               ls=':', lw=1.0, alpha=0.55)
    ax.set_ylabel('Pressure (Pa)', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)

axes_mb[-1].set_xlabel('Time relative to centroid muzzle blast (ms)', fontsize=11)
axes_mb[0].set_title(
    f'Friedlander Muzzle Blast Signals -- {RANGE:.0f} m range\n'
    f'dP ~ {dP_mb_all[0]:.2f} Pa  |  t_pos ~ {t_pos_all[0]*1e3:.1f} ms  '
    f'(dotted line)  |  SNR = 20 dB',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('muzzle_blast_signals.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.8 — Combined Per-Sensor Signal (ideal, noiseless)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  A REAL ADC samples ONE continuous stream per channel. Both events are
#  superimposed on a SINGLE shared time axis (t = 0 at the moment the shot is
#  fired). This cell produces the pure, noiseless signal the bullet creates:
#
#      p_bullet(t) = p_N-wave(t)  +  p_Friedlander(t)
#
#  Noise, reverb, and ADC effects are added in Sections 6/7, not here.
#
#  Physics reused unchanged from Sections 5.1/5.3 (shockwave) and 5.5/5.6
#  (muzzle blast) -- only the windowing/superposition changes.

PRE_ROLL  = 0.010         # 10 ms before the first shockwave arrival
POST_ROLL = 0.150         # 150 ms after the last muzzle-blast arrival (covers Friedlander decay)

t0 = min(t_arr.min(), t_mb.min()) - PRE_ROLL
t1 = t_mb.max() + POST_ROLL
t_master = np.arange(t0, t1, 1.0 / FS)

print(f"Master ADC timeline : {t0:.4f} s  ->  {t1:.4f} s")
print(f"Samples             : {len(t_master)}  ({len(t_master)/FS:.3f} s @ {FS/1000:.0f} kHz)")

def nwave_waveform(t, t_arrive, b, M, c, L):
    """Deterministic N-wave only (no noise) -- for superposition."""
    T_N = nwave_duration(b, M, c, L)
    dP  = nwave_peak_pressure(b)
    p   = np.zeros(len(t))
    mask = (t >= t_arrive) & (t < t_arrive + T_N)
    tau  = t[mask] - t_arrive
    p[mask] = dP * (1.0 - 2.0 * tau / T_N)
    return p

def friedlander_waveform(t, t_arrive, r):
    """Deterministic Friedlander blast only (no noise) -- for superposition."""
    dP, t_pos = friedlander_params(r)
    p = np.zeros(len(t))
    mask = t >= t_arrive
    tau  = t[mask] - t_arrive
    p[mask] = dP * (1.0 - tau / t_pos) * np.exp(-tau / t_pos)
    return p

sw_only = []   # kept for reference plotting / detector ground truth
mb_only = []
ideal_signals = []   # p_bullet(t) = sw + mb, no noise yet

for i in range(4):
    sw = nwave_waveform(t_master, t_arr[i], b_all[i], M, C, L_BULLET)
    mb = friedlander_waveform(t_master, t_mb[i], r_mb_all[i])
    sw_only.append(sw); mb_only.append(mb)
    ideal_signals.append(sw + mb)

print(f"\n{'Mic':<5} {'SW arr (s)':>11} {'SW peak (Pa)':>13} {'MB arr (s)':>12} {'MB peak (Pa)':>13} {'Event gap (s)':>14}")
for i in range(4):
    dP_sw = nwave_peak_pressure(b_all[i])
    dP_mb, t_pos_i = friedlander_params(r_mb_all[i])
    gap = t_mb[i] - (t_arr[i] + nwave_duration(b_all[i], M, C, L_BULLET))
    print(f"M{i}    {t_arr[i]:>11.5f} {dP_sw:>13.3f} {t_mb[i]:>12.5f} {dP_mb:>13.3f} {gap:>14.4f}")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6.1 — Noise Floor, Reverberation & Colored-Noise Models                ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  The Section 5.8 signal is physically correct but acoustically "dry" -- a
#  real sensor never sees a pure N-wave/Friedlander pulse on a flat zero
#  baseline. Effects modeled here:
#
#  1. FLAT NOISE FLOOR -- continuous sensor self-noise + electronics, added
#     once across the whole timeline (used by the simple capture model).
#
#  2. REVERBERATION -- ground/structure reflections create a decaying echo
#     tail after each event, characterized by RT60 (time for -60 dB decay).
#     Modeled as a statistical room impulse response (Moorer-style):
#     exponentially-decaying filtered noise, convolved with the dry signal.
#
#  3. COLORED NOISE -- real background noise (wind, electronics, distant
#     traffic) is NOT flat white noise; it typically has more energy at low
#     frequencies. Measured slope ~ -2.46 in log-log PSD space from a real
#     gunshot recording (close to "brown" noise, slope -2).
#
#  Reference constants were MEASURED from a real recording (see Section 2.2).

NOISE_FLOOR_PA = 0.005    # simple model: continuous sensor self-noise (~48 dB SPL)

RT60         = 1.25    # room decay time constant (s) -- raise for "boomier" indoor spaces
NOISE_RMS_PA = 0.075   # background noise floor in Pa -- controls peak/noise ratio
NOISE_SLOPE  = -2.4    # PSD slope: 0=white, -1=pink, -2=brown

def make_room_ir(rt60, fs, predelay=0.0, duration=None, seed=None):
    """Statistical room impulse response: exponentially-decaying noise burst."""
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = rt60 * 1.2
    n = int(duration * fs)
    t = np.arange(n) / fs
    decay = 10 ** (-3.0 * t / rt60)          # -60 dB at t = rt60
    ir = decay * rng.standard_normal(n)
    ir[0] = 1.0                               # direct path (unit impulse)
    pre_n = int(predelay * fs)
    if pre_n > 0:
        ir = np.concatenate([np.zeros(pre_n), ir])
    return ir

def make_colored_noise(n, fs, slope=-2.0, rms=1.0, seed=None):
    """Noise with a target PSD slope (0=white, -1=pink, -2=brown) via FFT filtering."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    freqs[0] = freqs[1]                       # avoid divide-by-zero at DC
    X = X * (freqs ** (slope / 2.0))
    colored = np.fft.irfft(X, n=n)
    return colored / (np.sqrt(np.mean(colored**2)) + 1e-12) * rms

print(f"Noise/distortion model configured: flat floor={NOISE_FLOOR_PA} Pa, "
      f"RT60={RT60}s, colored noise={NOISE_RMS_PA} Pa (slope={NOISE_SLOPE})")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7.1 — Simple Flat-Noise-Floor ADC Capture                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
rng_adc = np.random.default_rng(seed=7)
combined_signals = []   # p_sensor(t) = p_bullet(t) + n(t), one continuous stream

for i in range(4):
    n = NOISE_FLOOR_PA * rng_adc.standard_normal(len(t_master))
    combined_signals.append(ideal_signals[i] + n)

print(f"Noise floor : {NOISE_FLOOR_PA} Pa  "
      f"({20*np.log10(NOISE_FLOOR_PA/20e-6):.1f} dB SPL)")
print("This is the input to the blind two-stage event detector (Section 8.3).")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7.2 — Combined Signal Plots (flat-noise-floor model)                   ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for i in range(4):
    ax = axes[i]
    ax.plot(t_master, combined_signals[i], color=COLORS[i], lw=0.6)
    ax.axvline(t_arr[i], color='steelblue',  ls='--', lw=1.2, alpha=0.85, label='Shockwave')
    ax.axvline(t_mb[i],  color='darkorange', ls='--', lw=1.2, alpha=0.85, label='Muzzle blast')
    ax.set_ylabel('Pressure (Pa)', fontsize=10)
    ax.set_title(f'M{i} -- full ADC capture', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
axes[-1].set_xlabel('Time since shot fired (s)', fontsize=11)
fig.suptitle(
    f'Combined per-sensor received signal -- Mach {M}, {RANGE:.0f} m range\n'
    f'Dt = {Dt:.3f} s between events  |  noise floor = {NOISE_FLOOR_PA} Pa',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('combined_full_capture.png', dpi=150, bbox_inches='tight')
plt.show()

# -- Zoomed insets: M0, shockwave region vs muzzle-blast region ---------------
fig2, axz = plt.subplots(1, 2, figsize=(14, 4.2))

WIN_SW = 0.003
mask_sw = (t_master >= t_arr[0]-WIN_SW) & (t_master <= t_arr[0]+WIN_SW)
axz[0].plot((t_master[mask_sw]-t_arr[0])*1e3, combined_signals[0][mask_sw], color=COLORS[0], lw=1.2)
axz[0].axvline(0, color='steelblue', ls='--', lw=1.2)
axz[0].axhline(0, color='black', lw=0.5, alpha=0.4)
axz[0].set_title('Zoom: shockwave (N-wave) -- M0', fontsize=10, fontweight='bold')
axz[0].set_xlabel('ms relative to SW arrival'); axz[0].set_ylabel('Pressure (Pa)')
axz[0].grid(True, alpha=0.3)

WIN_MB = 0.04
mask_mb = (t_master >= t_mb[0]-WIN_MB) & (t_master <= t_mb[0]+WIN_MB)
axz[1].plot((t_master[mask_mb]-t_mb[0])*1e3, combined_signals[0][mask_mb], color=COLORS[0], lw=1.0)
axz[1].axvline(0, color='darkorange', ls='--', lw=1.2)
axz[1].axhline(0, color='black', lw=0.5, alpha=0.4)
axz[1].set_title('Zoom: muzzle blast (Friedlander) -- M0', fontsize=10, fontweight='bold')
axz[1].set_xlabel('ms relative to MB arrival'); axz[1].set_ylabel('Pressure (Pa)')
axz[1].grid(True, alpha=0.3)

fig2.suptitle('Same channel (M0), two different time scales -- note the noise floor before each onset',
              fontsize=10)
plt.tight_layout()
plt.savefig('combined_zoom.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7.3 — Full Realism Model Applied (reverb + colored noise + mic/ADC response)║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Reuses the idealized sw+mb waveforms from Section 5.8 (`nwave_waveform`,
#  `friedlander_waveform`, `t_master`) -- only the environmental/hardware layer
#  changes. Physics (arrival times, amplitudes) is untouched.

realistic_signals_pa = []   # still in physical Pa units
for i in range(4):
    dry = nwave_waveform(t_master, t_arr[i], b_all[i], M, C, L_BULLET) + \
          friedlander_waveform(t_master, t_mb[i], r_mb_all[i])

    # 1) Reverberant tail via convolution with a synthetic room IR
    rir = make_room_ir(RT60, FS, seed=1000+i)
    wet = fftconvolve(dry, rir, mode='full')[:len(dry)]
    if np.abs(dry).max() > 0:
        wet *= np.abs(dry).max() / (np.abs(wet).max() + 1e-12)   # preserve peak

    # 2) Colored background noise
    noise = make_colored_noise(len(t_master), FS, slope=NOISE_SLOPE,
                                rms=NOISE_RMS_PA, seed=2000+i)

    # 3) Mic/ADC bandpass response (Section 2.1)
    combined = mic_bandpass(wet + noise, FS, lo=MIC_LO, hi=MIC_HI)
    realistic_signals_pa.append(combined)

# -- Calibrate Pa -> normalized ADC units by PEAK match (prevents clipping) ---
peak_pa = max(np.abs(s).max() for s in realistic_signals_pa)
PA_TO_NORM = REAL_PEAK_NORM / peak_pa

realistic_signals = [quantize(s * PA_TO_NORM, bits=BIT_DEPTH) for s in realistic_signals_pa]

# -- Validate against the real-recording targets (Section 2.2) ----------------
calib_center = 0.5 * (t_arr.mean() + t_mb.mean())          # quiet gap between events
mask_calib   = (t_master >= calib_center-0.010) & (t_master <= calib_center+0.010)
noise_rms    = np.sqrt(np.mean(realistic_signals[0][mask_calib]**2))
peak_norm    = np.abs(realistic_signals[0]).max()

print(f"Peak (normalized)     : {peak_norm:.4f}   target {REAL_PEAK_NORM:.4f}")
print(f"Noise floor (gap)     : {noise_rms:.6f}   target {REAL_NOISE_RMS:.6f}")
print(f"Peak/noise SNR        : {20*np.log10(peak_norm/noise_rms):.1f} dB   "
      f"(real: {20*np.log10(REAL_PEAK_NORM/REAL_NOISE_RMS):.1f} dB)")

# RT60 check: fit decay slope of the tail while it's still above the noise floor
mb_local_peak = np.argmax(np.abs(realistic_signals[0][(t_master>t_mb[0]-0.001)&(t_master<t_mb[0]+0.01)]))
mb_idx = np.searchsorted(t_master, t_mb[0]) + mb_local_peak
tail = realistic_signals[0][mb_idx:mb_idx+int(1.0*FS)]
win_e = int(0.005*FS)
t_pts, e_pts = [], []
for k in range(0, len(tail)-win_e, win_e):
    r = np.sqrt(np.mean(tail[k:k+win_e]**2))
    if r > 3*noise_rms:
        t_pts.append(k/FS); e_pts.append(20*np.log10(r))
if len(t_pts) > 5:
    from scipy.stats import linregress
    slope, *_ = linregress(t_pts, e_pts)
    print(f"Measured RT60         : {-60/slope:.2f} s   target {REAL_RT60} s")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7.4 — Realistic Signal Plots (vs Real Recording Reference)             ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=False)

win_mask = (t_master >= t_arr[0]-0.05) & (t_master <= t_mb[0]+0.6)
axes[0].plot((t_master[win_mask]-t_arr[0])*1000, realistic_signals[0][win_mask],
             color='#2196F3', lw=0.5)
axes[0].set_title('Simulated M0 -- realistic model (reverb + colored noise + sensor response)',
                   fontweight='bold', fontsize=11)
axes[0].set_xlabel('ms relative to shockwave arrival'); axes[0].set_ylabel('Amplitude (norm.)')
axes[0].grid(True, alpha=0.3)

mb_mask = (t_master >= t_mb[0]-0.01) & (t_master <= t_mb[0]+0.55)
axes[1].plot((t_master[mb_mask]-t_mb[0])*1000, realistic_signals[0][mb_mask],
             color='#F44336', lw=0.5)
axes[1].set_title('Zoom: muzzle blast + reverberant tail', fontweight='bold', fontsize=11)
axes[1].set_xlabel('ms relative to muzzle blast arrival'); axes[1].set_ylabel('Amplitude (norm.)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('realistic_combined_signal.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.1 — GCC-PHAT Algorithm & Shockwave TDOA                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Algorithm
#  ---------
#  1. FFT both signals with n = N (native resolution, no time-domain zero-pad).
#  2. PHAT-weighted cross-spectrum:  G(f) = X1(f).X2*(f) / |X1(f).X2*(f)|
#  3. TDOA = argmax_tau { IFFT(G, zero-padded to n*interp_factor)[tau] }
#     (zero-padding in the FREQUENCY domain before the inverse FFT is what
#     gives sub-sample TDOA resolution -- doing it in the time domain before
#     the forward FFT instead causes a lag-scaling bug.)
#
#  At fs = 100 kHz, x16 zero-padding -> effective resolution = 0.625 us.
#
#  Reference: Knapp & Carter (1976), "The generalised cross-correlation
#             method for estimation of time delay", IEEE Trans. ASSP.

def gcc_phat(x1, x2, fs, interp_factor=16, max_lag_s=None):
    n = len(x1)
    X1 = np.fft.rfft(x1, n=n)
    X2 = np.fft.rfft(x2, n=n)
    G  = X2 * np.conj(X1)              # x2 relative to x1 (positive tau = x2 arrives later)
    G /= (np.abs(G) + 1e-12)
    nf = n * interp_factor
    cc = np.fft.fftshift(np.fft.irfft(G, n=nf))   # interpolation happens HERE
    lags = np.arange(-nf//2, nf//2) / (fs * interp_factor)
    if max_lag_s is None:
        max_lag_s = 1.5 * L_ARRAY / C
    mask = np.abs(lags) <= max_lag_s
    idx  = np.argmax(cc[mask])
    return lags[mask][idx], cc, lags

INTERP = 16
dt_eff = 1e6 / (FS * INTERP)

print(f"GCC-PHAT  (x{INTERP} zero-padding -> {dt_eff:.3f} us resolution)")
print("-" * 64)
print(f"{'Pair':<8} {'True tau (us)':>14} {'Est tau (us)':>14} {'Error (us)':>13}")

tdoa_results = {}
for i in range(1, 4):
    tau_true          = t_arr[i] - t_arr[0]
    tau_est, cc, lags = gcc_phat(signals[0], signals[i], FS, interp_factor=INTERP)
    tdoa_results[i]   = (tau_true, tau_est, cc, lags)
    err = abs(tau_est - tau_true)
    print(f"  (0,{i})    {tau_true*1e6:>+14.3f}   {tau_est*1e6:>+14.3f}   "
          f"{err*1e6:>13.3f}  {'OK' if err<10e-6 else '!'}")

fig, axes_g = plt.subplots(1, 3, figsize=(15, 4.5))
WIN_GCC = 1.2e-3

for idx, i in enumerate(range(1, 4)):
    tau_true, tau_est, cc, lags = tdoa_results[i]
    ax = axes_g[idx]
    view = np.abs(lags) <= WIN_GCC
    ax.plot(lags[view]*1e6, cc[view], color='steelblue', lw=1.3)
    ax.axvline(tau_true*1e6, color='crimson',   ls='--', lw=2.0,
               label=f'True: {tau_true*1e6:+.2f} us')
    ax.axvline(tau_est*1e6,  color='limegreen', ls=':',  lw=2.0,
               label=f'Est : {tau_est*1e6:+.2f} us')
    ax.set_xlabel('Lag (us)', fontsize=10)
    ax.set_ylabel('GCC-PHAT', fontsize=10)
    ax.set_title(f'M0 <-> M{i}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

fig.suptitle(
    f'GCC-PHAT Cross-Correlations  '
    f'(x{INTERP} zero-padding, {dt_eff:.2f} us effective resolution)',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('gcc_phat.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.2 — GCC-PHAT: Muzzle Blast TDOA (side-by-side vs Shockwave)          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Note: MB TDOA patterns are completely different from SW TDOAs because
#  they encode different physics (spherical vs Mach-cone wavefront geometry).

print(f"GCC-PHAT -- muzzle blast  (x{INTERP}, {dt_eff:.3f} us resolution)")
print("-" * 66)
print(f"{'Pair':<8} {'True tau_MB (us)':>16} {'Est tau_MB (us)':>15} {'Error (us)':>12}")

tdoa_mb_results = {}
for i in range(1, 4):
    tau_true = t_mb[i] - t_mb[0]
    tau_est, cc_mb, lags_mb = gcc_phat(
        signals_mb[0], signals_mb[i], FS, interp_factor=INTERP)
    tdoa_mb_results[i] = (tau_true, tau_est, cc_mb, lags_mb)
    err = abs(tau_est - tau_true)
    print(f"  (0,{i})    {tau_true*1e6:>+16.3f}   {tau_est*1e6:>+15.3f}   "
          f"{err*1e6:>12.3f}  {'OK' if err < 30e-6 else '!'}")

fig, axes_cmp = plt.subplots(2, 3, figsize=(15, 8))
WIN_CMP = 1.5e-3

for idx, i in enumerate(range(1, 4)):
    tt_sw, te_sw, cc_sw, lg_sw = tdoa_results[i]
    ax = axes_cmp[0, idx]
    view = np.abs(lg_sw) <= WIN_CMP
    ax.plot(lg_sw[view]*1e6, cc_sw[view], color='steelblue', lw=1.3)
    ax.axvline(tt_sw*1e6, color='crimson',   ls='--', lw=1.8,
               label=f'True: {tt_sw*1e6:+.1f} us')
    ax.axvline(te_sw*1e6, color='limegreen', ls=':',  lw=1.8,
               label=f'Est : {te_sw*1e6:+.1f} us')
    ax.set_title(f'Shockwave -- M0<->M{i}', fontsize=10, fontweight='bold', color='steelblue')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel('Lag (us)', fontsize=9)
    ax.set_ylabel('GCC-PHAT', fontsize=9)

    tt_mb, te_mb, cc_mb, lg_mb = tdoa_mb_results[i]
    ax2 = axes_cmp[1, idx]
    view2 = np.abs(lg_mb) <= WIN_CMP
    ax2.plot(lg_mb[view2]*1e6, cc_mb[view2], color='darkorange', lw=1.3)
    ax2.axvline(tt_mb*1e6, color='crimson',   ls='--', lw=1.8,
                label=f'True: {tt_mb*1e6:+.1f} us')
    ax2.axvline(te_mb*1e6, color='limegreen', ls=':',  lw=1.8,
                label=f'Est : {te_mb*1e6:+.1f} us')
    ax2.set_title(f'Muzzle blast -- M0<->M{i}', fontsize=10,
                  fontweight='bold', color='darkorange')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Lag (us)', fontsize=9)
    ax2.set_ylabel('GCC-PHAT', fontsize=9)

fig.suptitle('GCC-PHAT: Shockwave (blue) vs Muzzle Blast (orange)\n'
             'Different TDOA patterns -> different encoded physics',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('gcc_phat_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.3 — Adaptive Two-Stage Event Detector (blind detection on ADC capture)║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Unlike Sections 8.1/8.2 (which assume you already know where each event is
#  and hand GCC-PHAT a pre-cut window), this detector runs blind on the full
#  Section 7.1 ADC capture (`combined_signals`) the way real hardware would.
#
#  BUG FIX 1 (functional): the muzzle-blast search window must be configurable
#  (dt_min, dt_max) rather than hardcoded -- a fixed [1.0s, 4.0s] window tuned
#  for a 1000 m scenario would NEVER find a close-range shot (e.g. the Ruger
#  recordings' ~70 ms gap).
#
#  BUG FIX 2 (off-by-one): `np.diff(above.astype(int))` returns +1 at index i
#  when the crossing happens at i+1, not i. Using the raw diff index directly
#  reads the amplitude ONE SAMPLE BEFORE the actual onset -- silently
#  corrupting dP_sw and T_N measurements.

def detect_events(signal, t_axis, fs, dt_min=0.01, dt_max=5.0):
    """
    Two-stage detector: sharp shockwave threshold crossing, then an
    energy-based search for the muzzle blast within [dt_min, dt_max] seconds
    afterward. Widen these bounds if the expected range is unknown.
    """
    n_noise = int(0.005*fs)
    noise_rms = np.sqrt(np.mean(signal[:n_noise]**2))

    thresh_sw = 10.0*noise_rms
    above = np.abs(signal) > thresh_sw
    crossings_sw = np.where(np.diff(above.astype(int)) > 0)[0]
    if len(crossings_sw) == 0:
        raise RuntimeError("SW not detected")
    sw_idx = crossings_sw[0] + 1                 # +1: diff index -> actual onset sample
    t_sw_onset = t_axis[sw_idx]

    win_end = min(len(signal)-1, sw_idx+int(0.010*fs))
    sw_win = signal[sw_idx:win_end]; t_sw_win = t_axis[sw_idx:win_end]
    dP_measured = sw_win[0]
    zc = np.where(np.diff(np.sign(sw_win)))[0]
    T_N_meas = 2.0*(t_sw_win[zc[0]]-t_sw_win[0]) if len(zc) >= 1 else np.nan

    t_search_start = t_sw_onset + dt_min
    t_search_end   = t_sw_onset + dt_max
    mask_search = (t_axis >= t_search_start) & (t_axis <= t_search_end)
    sig_search = signal[mask_search]; t_search = t_axis[mask_search]
    if len(sig_search) < 10:
        raise RuntimeError("Search window too narrow / out of range")

    win_e = max(1, int(0.001*fs))
    energy = np.array([np.sum(sig_search[k:k+win_e]**2)/win_e
                       for k in range(len(sig_search)-win_e)])
    above_mb = energy > 20.0*noise_rms**2
    crossings_mb = np.where(np.diff(above_mb.astype(int)) > 0)[0]
    if len(crossings_mb) == 0:
        raise RuntimeError("MB not detected in search window")
    mb_idx = crossings_mb[0] + 1                  # +1: same off-by-one fix
    t_mb_onset = t_search[mb_idx]

    mb_win_idx = np.searchsorted(t_axis, t_mb_onset)
    mb_win = signal[mb_win_idx:mb_win_idx+int(0.005*fs)]
    dP_mb_est = mb_win.max() if len(mb_win) > 0 else np.nan

    return t_sw_onset, t_mb_onset, dP_measured, T_N_meas, dP_mb_est, noise_rms

print("Detecting events on all 4 channels...")
t_sw_det=np.zeros(4); t_mb_det=np.zeros(4)
dP_sw_det=np.zeros(4); T_N_det=np.zeros(4); dP_mb_det=np.zeros(4)
for i in range(4):
    ts,tm,dp,tn,dpm,nr = detect_events(combined_signals[i], t_master, FS, dt_min=0.01, dt_max=5.0)
    t_sw_det[i],t_mb_det[i],dP_sw_det[i],T_N_det[i],dP_mb_det[i] = ts,tm,dp,tn,dpm
    print(f"  M{i}: SW={ts:.5f}s  MB={tm:.5f}s  dP_sw={dp:.2f}Pa  T_N={tn*1e3:.3f}ms")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.4 — Direction-Finding TDOA on Shared Reference Windows               ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  BUG FIX 3 (subtle): windows must be extracted around a SHARED reference
#  time (mic 0's detected onset) -- NOT each mic's own detected onset.
#  Centering each mic's window on its own onset silently removes the very
#  TDOA the direction-finding solve depends on, making GCC-PHAT always report
#  tau=0 regardless of the true bearing.

def extract_window(signal, t_axis, t_center, t_pre, t_post):
    mask = (t_axis >= t_center-t_pre) & (t_axis < t_center+t_post)
    return signal[mask]

# All 4 channels windowed on mic 0's onset (with enough margin for the other
# mics' true arrival spread, which is at most L_ARRAY/c ~ 875us)
sw_wins = [extract_window(combined_signals[i], t_master, t_sw_det[0], 0.003, 0.015) for i in range(4)]
mb_wins = [extract_window(combined_signals[i], t_master, t_mb_det[0], 0.006, 0.080) for i in range(4)]

delta_p = np.array([mic_pos[i]-mic_pos[0] for i in range(1,4)])

d_sw = np.array([gcc_phat(sw_wins[0], sw_wins[i], FS)[0]*C for i in range(1,4)])
k_vec_sw = np.linalg.solve(delta_p, d_sw)
k_hat_sw = k_vec_sw/np.linalg.norm(k_vec_sw)

d_mb = np.array([gcc_phat(mb_wins[0], mb_wins[i], FS)[0]*C for i in range(1,4)])
k_vec_mb = np.linalg.solve(delta_p, d_mb)
n_hat_shooter = -k_vec_mb/np.linalg.norm(k_vec_mb)

print(f"k_hat_SW      = {k_hat_sw}   |k|={np.linalg.norm(k_vec_sw):.4f}  (should be ~1.0)")
print(f"n_hat_shooter = {n_hat_shooter}")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.5 — TDOA on the Full Realism Model (realistic_signals)               ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Everything in Section 8.3/8.4 ran on `combined_signals` (Section 7.1) --
#  the master timeline with only a flat noise floor added. This repeats the
#  same blind detection + windowed GCC-PHAT on `realistic_signals` (Section
#  7.3), which additionally has reverb smearing, colored (brown-ish) noise,
#  mic bandpass filtering, and 16-bit quantization -- i.e. the signal your
#  ADC would actually deliver.
#
#  FINDING: `detect_events()` as written in Section 8.3 actually FAILS here
#  with "MB not detected in search window". Root cause: its muzzle-blast
#  stage looks for a *rising* edge in energy (np.diff(above_mb) > 0). But the
#  reverberant tail from the shockwave (RT60 ~ 1.25 s) keeps the energy above
#  threshold continuously from the start of the search window all the way
#  into the muzzle blast -- there is no falling-then-rising edge to find, so
#  the detector never fires. This is precisely the "verify your TDOA
#  pipeline still performs on smeared onsets" risk flagged after Section 7.4.
#
#  detect_events_robust() below adds one guard: if the search window is
#  *already* above threshold at t=0 (i.e. still ringing from the shockwave),
#  it falls back to the first sample as an (unreliable) onset instead of
#  raising -- so you can see how bad the resulting TDOA gets, rather than the
#  pipeline just silently crashing.

def detect_events_robust(signal, t_axis, fs, dt_min=0.01, dt_max=5.0):
    n_noise = int(0.005*fs)
    noise_rms = np.sqrt(np.mean(signal[:n_noise]**2))

    thresh_sw = 10.0*noise_rms
    above = np.abs(signal) > thresh_sw
    crossings_sw = np.where(np.diff(above.astype(int)) > 0)[0]
    if len(crossings_sw) == 0:
        raise RuntimeError("SW not detected")
    sw_idx = crossings_sw[0] + 1
    t_sw_onset = t_axis[sw_idx]

    win_end = min(len(signal)-1, sw_idx+int(0.010*fs))
    sw_win = signal[sw_idx:win_end]; t_sw_win = t_axis[sw_idx:win_end]
    dP_measured = sw_win[0]
    zc = np.where(np.diff(np.sign(sw_win)))[0]
    T_N_meas = 2.0*(t_sw_win[zc[0]]-t_sw_win[0]) if len(zc) >= 1 else np.nan

    t_search_start = t_sw_onset + dt_min
    t_search_end   = t_sw_onset + dt_max
    mask_search = (t_axis >= t_search_start) & (t_axis <= t_search_end)
    sig_search = signal[mask_search]; t_search = t_axis[mask_search]
    if len(sig_search) < 10:
        raise RuntimeError("Search window too narrow / out of range")

    win_e = max(1, int(0.001*fs))
    energy = np.array([np.sum(sig_search[k:k+win_e]**2)/win_e
                       for k in range(len(sig_search)-win_e)])
    above_mb = energy > 20.0*noise_rms**2
    mb_reliable = True
    if above_mb[0]:
        # still ringing from the shockwave reverb tail -- no clean rising edge
        mb_idx = 0
        mb_reliable = False
    else:
        crossings_mb = np.where(np.diff(above_mb.astype(int)) > 0)[0]
        if len(crossings_mb) == 0:
            raise RuntimeError("MB not detected in search window")
        mb_idx = crossings_mb[0] + 1
    t_mb_onset = t_search[mb_idx]

    mb_win_idx = np.searchsorted(t_axis, t_mb_onset)
    mb_win = signal[mb_win_idx:mb_win_idx+int(0.005*fs)]
    dP_mb_est = mb_win.max() if len(mb_win) > 0 else np.nan

    return t_sw_onset, t_mb_onset, dP_measured, T_N_meas, dP_mb_est, noise_rms, mb_reliable

print("Detecting events on realistic_signals (full realism model)...")
t_sw_det_r=np.zeros(4); t_mb_det_r=np.zeros(4)
dP_sw_det_r=np.zeros(4); T_N_det_r=np.zeros(4); dP_mb_det_r=np.zeros(4)
mb_reliable_flags = []
for i in range(4):
    ts,tm,dp,tn,dpm,nr,rel = detect_events_robust(realistic_signals[i], t_master, FS,
                                                   dt_min=0.01, dt_max=5.0)
    t_sw_det_r[i],t_mb_det_r[i],dP_sw_det_r[i],T_N_det_r[i],dP_mb_det_r[i] = ts,tm,dp,tn,dpm
    mb_reliable_flags.append(rel)
    flag = "" if rel else "  <-- UNRELIABLE (reverb still above threshold at search start)"
    print(f"  M{i}: SW={ts:.5f}s  MB={tm:.5f}s  dP_sw={dp:.4f}  T_N={tn*1e3:.3f}ms{flag}")

# Windows for GCC-PHAT, centred on mic 0's onset (same shared-reference rule
# as Section 8.4 -- centering each mic on its own onset would erase the TDOA)
sw_wins_r = [extract_window(realistic_signals[i], t_master, t_sw_det_r[0], 0.003, 0.015) for i in range(4)]
mb_wins_r = [extract_window(realistic_signals[i], t_master, t_mb_det_r[0], 0.006, 0.080) for i in range(4)]

d_sw_r = np.array([gcc_phat(sw_wins_r[0], sw_wins_r[i], FS)[0]*C for i in range(1,4)])
k_vec_sw_r = np.linalg.solve(delta_p, d_sw_r)
k_hat_sw_r = k_vec_sw_r/np.linalg.norm(k_vec_sw_r)

d_mb_r = np.array([gcc_phat(mb_wins_r[0], mb_wins_r[i], FS)[0]*C for i in range(1,4)])
k_vec_mb_r = np.linalg.solve(delta_p, d_mb_r)
n_hat_shooter_r = -k_vec_mb_r/np.linalg.norm(k_vec_mb_r)

# -- Compare TDOA accuracy: flat-noise-floor capture vs. full realism model ----
print(f"\n{'Pair':<8} {'True SW (us)':>13} {'Flat (us)':>11} {'Err (us)':>10} "
      f"{'Realistic (us)':>15} {'Err (us)':>10}")
for i in range(1, 4):
    tau_true_sw = (t_arr[i] - t_arr[0]) * 1e6
    tau_flat_sw = gcc_phat(sw_wins[0], sw_wins[i], FS)[0]*1e6
    tau_real_sw = gcc_phat(sw_wins_r[0], sw_wins_r[i], FS)[0]*1e6
    print(f"(0,{i})    {tau_true_sw:>13.3f} {tau_flat_sw:>11.3f} "
          f"{abs(tau_flat_sw-tau_true_sw):>10.3f} {tau_real_sw:>15.3f} "
          f"{abs(tau_real_sw-tau_true_sw):>10.3f}")

print(f"\n{'Pair':<8} {'True MB (us)':>13} {'Flat (us)':>11} {'Err (us)':>10} "
      f"{'Realistic (us)':>15} {'Err (us)':>10}")
for i in range(1, 4):
    tau_true_mb = (t_mb[i] - t_mb[0]) * 1e6
    tau_flat_mb = gcc_phat(mb_wins[0], mb_wins[i], FS)[0]*1e6
    tau_real_mb = gcc_phat(mb_wins_r[0], mb_wins_r[i], FS)[0]*1e6
    print(f"(0,{i})    {tau_true_mb:>13.3f} {tau_flat_mb:>11.3f} "
          f"{abs(tau_flat_mb-tau_true_mb):>10.3f} {tau_real_mb:>15.3f} "
          f"{abs(tau_real_mb-tau_true_mb):>10.3f}")

print(f"\nk_hat_SW      (flat)      = {k_hat_sw}")
print(f"k_hat_SW      (realistic) = {k_hat_sw_r}")
print(f"n_hat_shooter (flat)      = {n_hat_shooter}")
print(f"n_hat_shooter (realistic) = {n_hat_shooter_r}")
if not all(mb_reliable_flags):
    print("\nWARNING: at least one channel's MB onset was flagged unreliable above --")
    print("the realistic-model MB TDOA/DOA numbers here should not be trusted as-is.")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.6 — Matched-Filter (CFAR-whitened) Muzzle-Blast Detection            ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  FIX vs. the version tested interactively: the search window used to be
#  hardcoded to 0.4 s past the shockwave onset. Dt itself is range-dependent
#  (it grows roughly linearly with range) -- a fixed 0.4 s window silently
#  misses the muzzle blast entirely once Dt exceeds it (e.g. at 500 m,
#  Dt ~ 0.85 s, so the true event was never even inside the search window).
#  The window is now sized from the PILAR-V forward model using the assumed
#  RANGE from Section 4, with margin -- exactly the kind of thing a real
#  system would size from its expected engagement envelope, not a magic
#  constant.

def build_friedlander_template(t_pos, fs, n_periods=6):
    n = max(4, int(np.ceil(n_periods * t_pos * fs)))
    tau = np.arange(n) / fs
    tpl = (1 - tau/t_pos) * np.exp(-tau/t_pos)
    tpl = tpl - tpl.mean()
    return tpl / np.linalg.norm(tpl)

def local_rms_envelope(x, fs, win_s=0.010):
    n = max(1, int(win_s * fs))
    ms = fftconvolve(x**2, np.ones(n)/n, mode="same")
    return np.sqrt(np.maximum(ms, 1e-20))

def cfar_matched_filter_locate(seg, template, fs, env_win_s=0.010):
    """Whiten by local RMS envelope, then normalized-correlate against template."""
    env   = local_rms_envelope(seg, fs, env_win_s)
    seg_w = seg / env
    n     = len(template)
    corr  = fftconvolve(seg_w, template[::-1], mode="valid")
    energy = fftconvolve(seg_w**2, np.ones(n), mode="valid")
    norm_corr = corr / (np.sqrt(np.maximum(energy, 1e-20)) * np.linalg.norm(template) + 1e-12)
    peak = int(np.argmax(np.abs(norm_corr)))
    return peak, norm_corr[peak]

def expected_dt(range_guess, b, M, c):
    """Forward PILAR-V model: Dt = t_MB - t_SW for an assumed range/miss distance."""
    beta = np.sqrt(M**2 - 1.0)
    t_sw = (range_guess + b*beta) / (M*c)
    t_mb = np.sqrt(range_guess**2 + b**2) / c
    return t_mb - t_sw

Dt_expected   = expected_dt(RANGE, b_ref, M, C)
SEARCH_MARGIN = 1.5                                  # safety factor over the nominal Dt
search_span_s = max(0.4, Dt_expected * SEARCH_MARGIN)

t_pos_assumed = T_POS_REF * (RANGE / R_REF_MB) ** (1.0/3.0)   # from Section 4's assumed RANGE
template_mb   = build_friedlander_template(t_pos_assumed, FS)

print(f"Expected Dt (from assumed RANGE={RANGE:.0f} m) = {Dt_expected*1e3:.1f} ms")
print(f"Search window sized to {search_span_s*1e3:.0f} ms past SW onset "
      f"({SEARCH_MARGIN}x margin)  (template t_pos = {t_pos_assumed*1e3:.2f} ms)")

t_mb_cfar = np.zeros(4)
print(f"{'Mic':<5} {'True MB (s)':>12} {'Detected (s)':>13} {'Error (ms)':>11} {'Score':>8}")
for i in range(4):
    sw_idx  = np.searchsorted(t_master, t_sw_det_r[i])
    lo, hi  = sw_idx + int(0.005*FS), sw_idx + int(search_span_s*FS)
    hi      = min(hi, len(realistic_signals[i]))
    seg     = realistic_signals[i][lo:hi]
    peak, score = cfar_matched_filter_locate(seg, template_mb, FS)
    onset_idx   = lo + peak
    t_mb_cfar[i] = t_master[onset_idx]
    err_ms = (t_mb_cfar[i] - t_mb[i]) * 1e3
    print(f"  M{i}   {t_mb[i]:>12.5f}  {t_mb_cfar[i]:>13.5f}  {err_ms:>+11.3f}  {score:>8.3f}")

print(
    "\nHONEST RESULT: even with the search window bug fixed, errors are still "
    "tens of ms, not the sub-10 us this pipeline needs. That remaining gap is "
    "NOT the window bug -- it's the noise-model mismatch diagnosed below."
)

# -- Why matched filtering / CFAR / STA-LTA all still struggle -----------------
idx_mb = np.searchsorted(t_master, t_mb[0])
window = int(0.005 * FS)
rir0 = make_room_ir(RT60, FS, seed=1000)
dry0 = nwave_waveform(t_master, t_arr[0], b_all[0], M, C, L_BULLET) + \
       friedlander_waveform(t_master, t_mb[0], r_mb_all[0])
wet0 = fftconvolve(dry0, rir0, mode="full")[:len(dry0)]
if np.abs(dry0).max() > 0:
    wet0 *= np.abs(dry0).max() / (np.abs(wet0).max() + 1e-12)
reverb_tail_rms  = np.sqrt(np.mean(wet0[idx_mb-window:idx_mb+window]**2))
mb_peak_pa       = dP_mb_all[0]

print(f"\n  Event gap Dt                    : {Dt*1e3:.1f} ms")
print(f"  Configured reverb RT60          : {RT60} s")
print(f"  Signal+residual RMS near true")
print(f"    MB arrival (M0, +/-5ms)        : {reverb_tail_rms:.3f} Pa")
print(f"  True muzzle blast peak          : {mb_peak_pa:.3f} Pa")
print(f"  -> ratio                        : {reverb_tail_rms/mb_peak_pa:.2f}  "
      f"({20*np.log10(mb_peak_pa/reverb_tail_rms):.1f} dB)")
print(
    "\n  RT60 = 1.25 s was calibrated from a close-range INDOOR pistol recording "
    "(Section 2.2), reused as-is for this long-range OUTDOOR rifle scenario. "
    "Section 8.7 tests what happens once RT60 is set to an open-field value."
)


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.7 — Fixing the Noise Model: Open-Field RT60                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Test: is the residual error in Section 8.6 caused by the noise-model
#  mismatch (RT60 calibrated indoors, reused outdoors), or something else?
#  Rebuild realistic_signals with an open-field RT60 and re-run detection.

RT60_outdoor = 0.25   # open field: negligible reverberant reflections

realistic_signals_outdoor = []
for i in range(4):
    dry = nwave_waveform(t_master, t_arr[i], b_all[i], M, C, L_BULLET) + \
          friedlander_waveform(t_master, t_mb[i], r_mb_all[i])
    rir = make_room_ir(RT60_outdoor, FS, seed=1000+i)
    wet = fftconvolve(dry, rir, mode="full")[:len(dry)]
    if np.abs(dry).max() > 0:
        wet *= np.abs(dry).max() / (np.abs(wet).max() + 1e-12)
    noise = make_colored_noise(len(t_master), FS, slope=NOISE_SLOPE, rms=NOISE_RMS_PA, seed=2000+i)
    combined = mic_bandpass(wet + noise, FS, lo=MIC_LO, hi=MIC_HI)
    realistic_signals_outdoor.append(combined)

peak_pa_o = max(np.abs(s).max() for s in realistic_signals_outdoor)
PA_TO_NORM_o = REAL_PEAK_NORM / peak_pa_o
realistic_signals_outdoor = [quantize(s * PA_TO_NORM_o, bits=BIT_DEPTH) for s in realistic_signals_outdoor]

print(f"RT60 = {RT60_outdoor} s (open field) instead of {RT60} s (indoor).  "
      f"Dt (true event gap) = {Dt*1e3:.1f} ms\n")

# -- Attempt 1: original dt_min=0.01s -------------------------------------------
print("(1) Section 8.3 detector, dt_min = 0.01 s (unchanged):")
for i in range(4):
    try:
        ts, tm, dp, tn, dpm, nr = detect_events(realistic_signals_outdoor[i], t_master, FS,
                                                 dt_min=0.01, dt_max=5.0)
        print(f"  M{i}: MB={tm:.5f}s  error = {(tm-t_mb[i])*1e3:+.2f} ms")
    except RuntimeError as e:
        print(f"  M{i}: FAILED -- {e}")

# -- FINDING: even at RT60=0.25s, the SW's own reverb tail crosses the -----------
# detection threshold right around t ~ 1.1 x RT60 after the SW onset (verified
# by scanning the residual energy profile directly: it decays roughly on the
# RT60 schedule, ~4 Pa near t=0 down to ~0.01 Pa by ~260 ms). A fixed
# dt_min=0.01s lets the detector lock onto that decaying tail's threshold
# crossing instead of waiting for it to fully die down.

dt_min_scaled = 1.1 * RT60_outdoor
print(f"\n(2) dt_min scaled to 1.1x RT60 = {dt_min_scaled*1e3:.0f} ms:")
if dt_min_scaled >= Dt:
    print("  WARNING: dt_min >= Dt -- RT60 exceeds the event gap itself; no dt_min works.")
else:
    t_mb_scaled = np.zeros(4)
    for i in range(4):
        ts, tm, dp, tn, dpm, nr = detect_events(realistic_signals_outdoor[i], t_master, FS,
                                                 dt_min=dt_min_scaled, dt_max=5.0)
        t_mb_scaled[i] = tm
        print(f"  M{i}: MB={tm:.5f}s  error = {(tm-t_mb[i])*1e3:+.2f} ms")

    # -- Attempt to refine with GCC-PHAT on shared-reference windows -----------
    mb_wins_scaled = [extract_window(realistic_signals_outdoor[i], t_master, t_mb_scaled[0],
                                      0.006, 0.080) for i in range(4)]
    print(f"\n  Refining via GCC-PHAT (shared-reference windows, Section 8.4 style):")
    print(f"  {'Pair':<8} {'True MB (us)':>13} {'Est MB (us)':>12} {'Error (us)':>11}")
    for i in range(1, 4):
        tau_true = (t_mb[i] - t_mb[0]) * 1e6
        tau_est  = gcc_phat(mb_wins_scaled[0], mb_wins_scaled[i], FS)[0] * 1e6
        print(f"  (0,{i})   {tau_true:>13.3f}  {tau_est:>12.3f}  {abs(tau_est-tau_true):>11.3f}")

print(
    "\nSCORECARD so far:"
    "\n  RT60=1.25s, dt_min=0.01s   (Sec. 8.3/8.5 as shipped): FAILS -- detector never fires"
    "\n  RT60=0.25s, dt_min=0.01s   (this cell, attempt 1)    : ~150-200 ms error"
    "\n  RT60=0.25s, dt_min=1.1xRT60 (this cell, attempt 2)   : ~15-20 ms error (10x better)"
    "\n  + GCC-PHAT refinement on top of that                 : still tens-hundreds of us off"
    "\n"
    "\nSetting dt_min ~ RT60 fixes the *coarse* onset (10x improvement) by waiting "
    "out the shockwave's own reverb tail before searching. But the coarse onset "
    "still lands ~15-20 ms early/late per channel in a way that ISN'T a simple "
    "shared/common bias -- windowing GCC-PHAT around it doesn't recover accurate "
    "TDOA. Getting to the sub-10 us this pipeline needs will take a proper "
    "reverberation-robust refinement stage (e.g. deconvolving/whitening against "
    "the room response before correlating, not just picking a better dt_min), "
    "which is a real next piece of work rather than a parameter tweak."
)


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.8 — Frequency-Domain Dereverberation (Wiener Deconvolution)          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Frequency-domain dereverberation via regularized inverse filtering (Wiener
#  deconvolution). If Y(f) = X(f).H(f) + noise, the regularized inverse is:
#
#      X_hat(f) = Y(f) . H*(f) / (|H(f)|^2 + reg . max|H(f)|^2)
#
#  IMPORTANT CAVEAT: this uses the EXACT room impulse response used to build
#  the signal in Section 7.3/8.7 -- ground truth a real deployed system would
#  NOT have (it would only know statistics like RT60, not this specific
#  random realization). Treat this as an UPPER-BOUND sanity check answering
#  "is removing the reverb even capable of fixing this in principle?" before
#  investing in blind dereverberation (estimating the RIR/its statistics
#  from the recording itself -- harder, and covered in the notes below).

def wiener_deconvolve(y, rir, reg=1e-2):
    """Regularized frequency-domain inverse filter (a simple Wiener deconvolution)."""
    n = len(y)
    n_fft = n + len(rir) - 1
    Y = np.fft.rfft(y, n=n_fft)
    H = np.fft.rfft(rir, n=n_fft)
    H_mag2 = np.abs(H) ** 2
    G = np.conj(H) / (H_mag2 + reg * np.max(H_mag2))
    x_hat = np.fft.irfft(Y * G, n=n_fft)
    return x_hat[:n]

def build_deconvolved_signals(rt60_value, reg=1e-2):
    """Rebuild the realistic signal chain, but deconvolve each channel with
    its own exact room IR before mic bandpass + quantization."""
    out = []
    for i in range(4):
        rir_i = make_room_ir(rt60_value, FS, seed=1000 + i)
        dry_i = nwave_waveform(t_master, t_arr[i], b_all[i], M, C, L_BULLET) + \
                friedlander_waveform(t_master, t_mb[i], r_mb_all[i])
        wet_i = fftconvolve(dry_i, rir_i, mode="full")[:len(dry_i)]
        if np.abs(dry_i).max() > 0:
            wet_i *= np.abs(dry_i).max() / (np.abs(wet_i).max() + 1e-12)
        noise_i = make_colored_noise(len(t_master), FS, slope=NOISE_SLOPE, rms=NOISE_RMS_PA, seed=2000+i)
        received_i = wet_i + noise_i
        deconv_i = wiener_deconvolve(received_i, rir_i, reg=reg)
        out.append(mic_bandpass(deconv_i, FS, lo=MIC_LO, hi=MIC_HI))
    peak_pa = max(np.abs(s).max() for s in out)
    return [quantize(s * (REAL_PEAK_NORM / peak_pa), bits=BIT_DEPTH) for s in out]

for rt60_label, rt60_value in [("RT60 = 1.25 s (notebook default, indoor)", RT60),
                                 ("RT60 = 0.25 s (open field)", RT60_outdoor)]:
    print(f"--- {rt60_label} ---")
    deconv_signals = build_deconvolved_signals(rt60_value)

    print("Section 8.3 detector, dt_min=0.01s -- UNCHANGED from what shipped originally:")
    t_sw_dc = np.zeros(4); t_mb_dc = np.zeros(4); ok = True
    for i in range(4):
        try:
            ts, tm, dp, tn, dpm, nr = detect_events(deconv_signals[i], t_master, FS, dt_min=0.01, dt_max=5.0)
            t_sw_dc[i], t_mb_dc[i] = ts, tm
            print(f"  M{i}: MB error = {(tm-t_mb[i])*1e3:+.3f} ms")
        except RuntimeError as e:
            print(f"  M{i}: FAILED -- {e}"); ok = False

    if ok:
        mb_wins_dc = [extract_window(deconv_signals[i], t_master, t_mb_dc[0], 0.006, 0.080) for i in range(4)]
        print("  GCC-PHAT on shared-reference windows:")
        for i in range(1, 4):
            tau_true = (t_mb[i] - t_mb[0]) * 1e6
            tau_est  = gcc_phat(mb_wins_dc[0], mb_wins_dc[i], FS)[0] * 1e6
            print(f"    (0,{i})  true={tau_true:>+9.3f} us  est={tau_est:>+9.3f} us  "
                  f"err={abs(tau_est-tau_true):>7.3f} us")
    print()

print(
    "RESULT: with the EXACT room response removed, GCC-PHAT recovers sub-10 us "
    "TDOA accuracy in BOTH the indoor (RT60=1.25s) and outdoor (RT60=0.25s) "
    "cases -- essentially matching the flat-noise-floor pipeline. This confirms "
    "reverberation, not colored noise or the mic/ADC model, was the entire "
    "blocker."
    "\n\n"
    "The catch, again: this deconvolution used the exact RIR realization "
    "(ground truth). A real system doesn't have that -- it would need BLIND "
    "dereverberation, estimating either the RIR itself (e.g. from repeated "
    "shots / a calibration chirp) or its statistics (RT60, spectral envelope) "
    "and using a technique like Weighted Prediction Error (WPE) linear "
    "prediction dereverberation, which doesn't require knowing the exact "
    "impulse response. That's the real next step if this matters for your "
    "deployment -- happy to prototype a blind version if useful."
)


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8.9 — Blind Dereverberation (WPE): Does It Actually Work Here?         ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  Section 8.8 used the EXACT room response (ground truth). This tests what
#  you actually get WITHOUT that: a real blind dereverberation method, WPE
#  (Weighted Prediction Error) -- the standard technique for exactly this
#  problem in speech processing. WPE estimates a linear predictor of the
#  "late reverberation" from the recording's own past STFT frames, per
#  frequency bin, with no knowledge of the room impulse response:
#
#      Y(f,t) = D(f,t) + sum_{k=1}^{K} g(f,k) . Y(f, t-delay-k+1)
#
#  solved via iteratively-reweighted least squares (weights ~ 1/|D(f,t)|^2).

from scipy.signal import stft, istft

def wpe_single_channel(y, fs, n_fft=512, hop=128, delay=2, taps=10, n_iter=3):
    f, t, Y = stft(y, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
    n_freq, n_frames = Y.shape
    D = Y.copy()
    for _ in range(n_iter):
        weights = 1.0 / (np.abs(D)**2 + 1e-8)
        D_new = Y.copy()
        for fi in range(n_freq):
            valid_t = np.arange(delay+taps-1, n_frames)
            if len(valid_t) < taps + 5:
                continue
            X = np.zeros((len(valid_t), taps), dtype=complex)
            for k in range(taps):
                X[:, k] = Y[fi, valid_t-delay-k]
            w = weights[fi, valid_t]
            Xw = X * w[:, None]
            A = Xw.conj().T @ X
            b = Xw.conj().T @ Y[fi, valid_t]
            try:
                g = np.linalg.solve(A + 1e-6*np.eye(taps), b)
            except np.linalg.LinAlgError:
                continue
            D_new[fi, valid_t] = Y[fi, valid_t] - X @ g
        D = D_new
    _, d = istft(D, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
    if len(d) < len(y):
        d = np.pad(d, (0, len(y)-len(d)))
    return d[:len(y)]

print("Running blind WPE on all 4 channels (no RIR, no RT60 used)...")
wpe_signals_pa = []
for i in range(4):
    rir_i      = make_room_ir(RT60_outdoor, FS, seed=1000+i)
    dry_i      = nwave_waveform(t_master, t_arr[i], b_all[i], M, C, L_BULLET) + \
                 friedlander_waveform(t_master, t_mb[i], r_mb_all[i])
    wet_i      = fftconvolve(dry_i, rir_i, mode="full")[:len(dry_i)]
    if np.abs(dry_i).max() > 0:
        wet_i *= np.abs(dry_i).max() / (np.abs(wet_i).max() + 1e-12)
    noise_i    = make_colored_noise(len(t_master), FS, slope=NOISE_SLOPE, rms=NOISE_RMS_PA, seed=2000+i)
    received_i = wet_i + noise_i
    wpe_i      = wpe_single_channel(received_i, FS, n_fft=512, hop=128, delay=2, taps=10, n_iter=3)
    wpe_signals_pa.append(mic_bandpass(wpe_i, FS, lo=MIC_LO, hi=MIC_HI))

peak_pa_wpe = max(np.abs(s).max() for s in wpe_signals_pa)
wpe_signals = [quantize(s * (REAL_PEAK_NORM/peak_pa_wpe), bits=BIT_DEPTH) for s in wpe_signals_pa]

print("\nSection 8.3 detector, dt_min=0.01s:")
t_mb_wpe = np.zeros(4)
for i in range(4):
    ts, tm, dp, tn, dpm, nr = detect_events(wpe_signals[i], t_master, FS, dt_min=0.01, dt_max=5.0)
    t_mb_wpe[i] = tm
    print(f"  M{i}: MB error = {(tm-t_mb[i])*1e3:+.3f} ms")

dt_min_scaled_wpe = 1.1 * RT60_outdoor
print(f"\ndt_min scaled to 1.1xRT60 = {dt_min_scaled_wpe*1e3:.0f} ms:")
t_mb_wpe2 = np.zeros(4)
for i in range(4):
    ts, tm, dp, tn, dpm, nr = detect_events(wpe_signals[i], t_master, FS, dt_min=dt_min_scaled_wpe, dt_max=5.0)
    t_mb_wpe2[i] = tm
    print(f"  M{i}: MB error = {(tm-t_mb[i])*1e3:+.3f} ms")

mb_wins_wpe = [extract_window(wpe_signals[i], t_master, t_mb_wpe2[0], 0.006, 0.080) for i in range(4)]
print("\nGCC-PHAT on shared-reference windows:")
for i in range(1, 4):
    tau_true = (t_mb[i] - t_mb[0]) * 1e6
    tau_est  = gcc_phat(mb_wins_wpe[0], mb_wins_wpe[i], FS)[0] * 1e6
    print(f"  (0,{i})  true={tau_true:>+9.3f} us  est={tau_est:>+9.3f} us  "
          f"err={abs(tau_est-tau_true):>8.3f} us")

print(
    "\nHONEST RESULT: blind WPE gives essentially NO improvement here -- errors "
    "are the same as running no dereverberation at all (compare to Section 8.7's "
    "attempt-1 numbers). This is a real negative result, not a tuning failure: "
    "WPE's linear predictor needs many overlapping-reverberation frames to "
    "estimate g(f,k) reliably (that's why it works well on continuous speech). "
    "This recording is mostly silence with one brief impulsive event -- there "
    "simply isn't enough reverberant excitation in a single shot for a blind, "
    "per-shot statistical estimate to learn a usable predictor."
    "\n\n"
    "What WOULD work in a real deployment: this is a FIXED sensor location, so "
    "the room response doesn't change shot-to-shot. Measure it ONCE with a "
    "calibration signal (a test shot, clap, or swept chirp) at install time, "
    "store that impulse response, and reuse the Section 8.8 exact-deconvolution "
    "approach for every subsequent live detection -- turning the 'ground truth "
    "we don't have' problem into a one-time calibration step, which is exactly "
    "how real acoustic gunshot-localization systems handle site-specific "
    "reverberation."
)


from scipy.optimize import brentq

# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9.1 — Three Velocity Estimation Methods (A/B/C)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  METHOD A -- Mach angle only:  k_hat_SW . v_hat ~ 1/M, using v_hat ~
#  -n_hat_shooter as a cheap approximation of bullet direction. Biased by the
#  miss distance: the true error in 1/M is ~ beta.(b/R). No calibration
#  needed, but least accurate.
#
#  METHOD B -- N-wave duration + amplitude. Requires calibrated sensors (dP in
#  physical Pa) and known bullet length L. Solves two Whitham-scaling
#  equations (duration, amplitude) simultaneously for M -- most accurate here
#  because it doesn't depend on the noisy geometric approximation in A.
#
#  METHOD C -- Geometric correction: uses the miss distance b (from Method B's
#  amplitude fit) and a locally-solved PILAR-V range to correct v_hat for the
#  actual trajectory angle, rather than approximating v_hat ~ -n_hat_shooter.
#  This range solve is self-contained here and independent of Section 10's
#  official range/location estimate.
#
#  L_BULLET, dP0_sw, b0_sw already come from the BULLET_LIBRARY entry chosen
#  in Section 3.1 -- not redefined here.

# -- METHOD A ------------------------------------------------------------------
v_hat_approx = -n_hat_shooter
dot_A = np.dot(k_hat_sw, v_hat_approx)
M_A = 1.0/dot_A if dot_A > 0 else float('nan')

# -- METHOD B ------------------------------------------------------------------
M_B_estimates = []
for i in range(4):
    dP, T_N = abs(dP_sw_det[i]), T_N_det[i]
    if np.isnan(T_N) or dP < 0.05:
        continue
    b_est = b0_sw*(dP0_sw/dP)**2
    def f_M(M):
        if M <= 1.0: return -1e10
        return T_N**2*M**2*C**2*np.sqrt(M**2-1) - 4.0*b_est*L_BULLET
    if f_M(1.001)*f_M(10.0) > 0:
        continue
    M_B_estimates.append(brentq(f_M, 1.001, 10.0))
M_B = np.mean(M_B_estimates)

# -- METHOD C ------------------------------------------------------------------
def pilar_v_exact(Dt_val, b, M, c, R_max=5000.0):
    beta = np.sqrt(M**2-1.0)
    def f(R): return np.sqrt(R**2+b**2)/c - (R+b*beta)/(M*c) - Dt_val
    return brentq(f, 0.1, R_max)

b_corr = b0_sw*(dP0_sw/abs(dP_sw_det.mean()))**2
Dt_measured = t_mb_det.mean() - t_sw_det.mean()
R_est = pilar_v_exact(Dt_measured, b_corr, M_B, C)
sin_alpha = b_corr/np.sqrt(R_est**2+b_corr**2)
k_sw_along = np.dot(k_hat_sw, n_hat_shooter)*n_hat_shooter
e_lat = -(k_hat_sw - k_sw_along); e_lat /= np.linalg.norm(e_lat)
cos_alpha = np.sqrt(1-sin_alpha**2)
v_hat_corr = cos_alpha*(-n_hat_shooter) + sin_alpha*e_lat
v_hat_corr /= np.linalg.norm(v_hat_corr)
dot_C = np.dot(k_hat_sw, v_hat_corr)
M_C = 1.0/dot_C if dot_C > 0 else float('nan')

# -- Summary -------------------------------------------------------------------
print("="*58)
print(f"{'Method':<28} {'Mach':>8} {'V (m/s)':>10}")
print("-"*58)
print(f"{'A: angle only':<28} {M_A:>8.3f} {M_A*C:>10.1f}")
print(f"{'B: duration+amplitude':<28} {M_B:>8.3f} {M_B*C:>10.1f}")
print(f"{'C: geometric correction':<28} {M_C:>8.3f} {M_C*C:>10.1f}")
print("="*58)
print(f"Estimated range (PILAR-V, local to this cell): {R_est:.1f} m   miss distance: {b_corr:.1f} m")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10.1 — Shooter DOA from Muzzle Blast TDOAs (Plane-wave Model)          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  PLANE-WAVE APPROXIMATION  (valid when range >> aperture)
#  ------------------------------------------------------------------------------
#  Muzzle blast TDOAs for mic i relative to reference mic 0:
#      c . tau_i0 ~ (p_i - p_0) . k_hat      where k_hat = unit wave propagation direction
#
#  Three pairs (i = 1, 2, 3) -> exactly-determined 3x3 linear system:
#      delta_p . k_hat = d             delta_p_i = p_i - p_0,   d_i = c . tau_i0
#
#  Solve -> k_hat_est (normalise) -> shooter direction n_hat_src = -k_hat
#
#  From n_hat_src:
#      Azimuth   phi   = arctan2(n_y, n_x)
#      Elevation theta = arcsin(n_z)
#
#  Uses the "known event windows" GCC-PHAT results from Section 8.2
#  (`tdoa_mb_results`), not the blind detector from Section 8.3/8.4.

delta_p    = np.array([mic_pos[i] - mic_pos[0] for i in range(1, 4)])   # (3, 3)

d_mb_true  = np.array([(t_mb[i]              - t_mb[0])  * C for i in range(1, 4)])
d_mb_est   = np.array([tdoa_mb_results[i][1]             * C for i in range(1, 4)])

k_vec_true = np.linalg.solve(delta_p, d_mb_true)
k_vec_est  = np.linalg.solve(delta_p, d_mb_est)

n_src_true   = -k_vec_true / np.linalg.norm(k_vec_true)
n_src_est    = -k_vec_est  / np.linalg.norm(k_vec_est)
n_src_actual =  SHOOTER_POS / np.linalg.norm(SHOOTER_POS)

def az_el(n):
    az = np.degrees(np.arctan2(n[1], n[0]))
    el = np.degrees(np.arcsin(np.clip(n[2], -1.0, 1.0)))
    return az, el

az_act, el_act = az_el(n_src_actual)
az_tru, el_tru = az_el(n_src_true)
az_est, el_est = az_el(n_src_est)

print("Shooter DOA from muzzle blast TDOAs")
print("-" * 56)
print(f"  {'':24} {'Azimuth (deg)':>13} {'Elevation (deg)':>16}")
print(f"  Actual shooter         {az_act:>13.3f}   {el_act:>16.3f}")
print(f"  True TDOAs -> DOA      {az_tru:>13.3f}   {el_tru:>16.3f}")
print(f"  GCC-PHAT -> DOA        {az_est:>13.3f}   {el_est:>16.3f}")
az_err = abs(az_est - az_act)
el_err = abs(el_est - el_act)
print(f"  DOA error (GCC-PHAT)   {az_err:>13.3f} deg  {el_err:>16.3f} deg")
print(f"\n  Propagation direction k_hat (estimated): "
      f"[{-n_src_est[0]:.4f}, {-n_src_est[1]:.4f}, {-n_src_est[2]:.4f}]")
print(f"  |k_hat| = {np.linalg.norm(k_vec_est):.6f}  (should be ~ 1.0)")


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10.2 — PILAR-V Range Estimation + 3D Localization                      ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#  PILAR-V FORMULA  (Valiere et al., EURONOISE 2006; METRAVIB patent)
#  ------------------------------------------------------------------------------
#  Simplified geometry (shooter at (-R, 0), bullet flies in +x, miss dist. b):
#      t_SW = (R + b.beta) / (M.c)         Mach cone (Section 5.1)
#      t_MB = sqrt(R^2 + b^2) / c            spherical spreading
#      Dt = t_MB - t_SW = sqrt(R^2+b^2)/c - (R+b.beta)/(M.c)
#
#  Exact range: solve numerically for R given {Dt, b, M, c}.
#  Approx range (b << R):  R_hat ~ (Dt . M . c + b . beta) / (M - 1)
#
#  Range sensitivity:  dR/dDt ~ 515 m/s at M=3, R=1000 m, b=50 m
#      -> 1 ms timing error -> 0.515 m range error

from scipy.optimize import brentq

def pilar_v_exact(Dt_val, b, M, c, R_max=5000.0):
    """Exact PILAR-V range (along-track) from Dt via brentq root-finding."""
    beta = np.sqrt(M**2 - 1.0)
    def f(R): return np.sqrt(R**2 + b**2)/c - (R + b*beta)/(M*c) - Dt_val
    return brentq(f, 0.1, R_max)

def pilar_v_approx(Dt_val, b, M, c):
    """Approximate formula valid when b << R."""
    beta = np.sqrt(M**2 - 1.0)
    return (Dt_val * M * c + b * beta) / (M - 1.0)

# -- Range estimates ------------------------------------------------------------
b_meas   = b_ref      # perpendicular miss distance from Section 5.1
Dt_true  = Dt         # from Section 5.5 (exact timing)

rng_dt   = np.random.default_rng(seed=17)
Dt_noise = rng_dt.normal(0, 0.5e-3)
Dt_est   = Dt_true + Dt_noise

R_along_actual = a_ref                              # true along-track
R_3D_actual    = np.linalg.norm(SHOOTER_POS)        # true 3D distance

R_approx       = pilar_v_approx(Dt_true, b_meas, M, C)
R_exact_true   = pilar_v_exact( Dt_true, b_meas, M, C)
R_exact_est    = pilar_v_exact( Dt_est,  b_meas, M, C)

print("PILAR-V Range Estimation")
print("-" * 60)
print(f"  Dt (true)             : {Dt_true*1e3:.4f} ms")
print(f"  Dt (estimated + noise): {Dt_est*1e3:.4f} ms  "
      f"(added {Dt_noise*1e3:+.2f} ms jitter)")
print(f"  b (miss distance)     : {b_meas:.4f} m")
print(f"  R_along  actual       : {R_along_actual:.2f} m  (along-track)")
print(f"  R_3D     actual       : {R_3D_actual:.2f} m  (3D distance)")
print()
print(f"  Approx formula (Dt_true)  : {R_approx:.2f} m  "
      f"(delta = {R_approx - R_along_actual:+.2f} m vs along-track)")
print(f"  Exact  formula (Dt_true)  : {R_exact_true:.2f} m  "
      f"(delta = {R_exact_true - R_along_actual:+.2f} m)")
print(f"  Exact  formula (Dt + noise): {R_exact_est:.2f} m  "
      f"(delta = {R_exact_est - R_along_actual:+.2f} m)")

eps     = 1e-5
dR_dDt  = (pilar_v_exact(Dt_true+eps, b_meas, M, C) -
            pilar_v_exact(Dt_true-eps, b_meas, M, C)) / (2*eps)
print(f"\n  dR/dDt ~ {dR_dDt:.1f} m/s")
print(f"  -> 1.0 ms error -> {abs(dR_dDt*1e-3):.3f} m range error")
print(f"  -> 1 sample at {FS//1000} kHz (10 us) -> {abs(dR_dDt*1e-5):.4f} m range error")

# -- 3D position estimate --------------------------------------------------------
R_3D_est        = np.sqrt(R_exact_est**2 + b_meas**2)
shooter_est_pos = R_3D_est * n_src_est    # n_src_est from Section 10.1
err_3D          = np.linalg.norm(shooter_est_pos - SHOOTER_POS)

print(f"\n  3D position estimate : [{shooter_est_pos[0]:+.1f}, "
      f"{shooter_est_pos[1]:+.1f}, {shooter_est_pos[2]:+.1f}] m")
print(f"  Actual position      : [{SHOOTER_POS[0]:+.1f}, "
      f"{SHOOTER_POS[1]:+.1f}, {SHOOTER_POS[2]:+.1f}] m")
print(f"  3D localisation error: {err_3D:.2f} m")

# -- Plots ------------------------------------------------------------------------
fig, axes_pv = plt.subplots(1, 2, figsize=(14, 5))

Dt_sweep    = np.linspace(Dt_true*0.9, Dt_true*1.1, 300)
R_sweep     = [pilar_v_exact(dt, b_meas, M, C) for dt in Dt_sweep]
Dt_err_ms   = (Dt_sweep - Dt_true) * 1e3
R_err_m     = np.array(R_sweep) - R_along_actual

ax = axes_pv[0]
ax.plot(Dt_err_ms, R_err_m, color='steelblue', lw=2)
ax.axvline(0, color='black', lw=1.0, alpha=0.4)
ax.axhline(0, color='black', lw=1.0, alpha=0.4)
ax.axvline(Dt_noise*1e3, color='crimson', ls='--', lw=1.8,
           label=f'This run: Dt err = {Dt_noise*1e3:+.2f} ms\n'
                 f'-> DeltaR = {R_exact_est-R_along_actual:+.2f} m')
slope_txt = f'Slope ~ {dR_dDt*1e-3:.3f} m/ms'
ax.text(0.05, 0.9, slope_txt, transform=ax.transAxes,
        fontsize=10, color='steelblue')
ax.set_xlabel('Dt estimation error (ms)', fontsize=11)
ax.set_ylabel('Range error (m)', fontsize=11)
ax.set_title('PILAR-V Range Sensitivity\ndR/dDt ~ {:.0f} m/s'.format(dR_dDt),
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

ax2 = axes_pv[1]
ax2.scatter(0, 0, color='black', s=150, marker='^', zorder=8,
            label='Array centroid')
for p, col in zip(mic_pos, COLORS):
    ax2.scatter(p[0], p[1], color=col, s=60, zorder=8)
ax2.plot([BULLET_ORIGIN[0], 100], [Y_MISS, Y_MISS],
         '-', color='darkorange', lw=1.8, alpha=0.7, label='Bullet path')
ax2.scatter(SHOOTER_POS[0], SHOOTER_POS[1], color='darkred',
            s=220, marker='*', zorder=9,
            label=f'Actual ({SHOOTER_POS[0]:.0f}, {SHOOTER_POS[1]:.0f}) m')
ax2.scatter(shooter_est_pos[0], shooter_est_pos[1], color='limegreen',
            s=160, marker='D', zorder=9,
            label=f'Estimated ({shooter_est_pos[0]:.0f}, {shooter_est_pos[1]:.0f}) m')
ax2.plot([0, SHOOTER_POS[0]],    [0, SHOOTER_POS[1]],    'r--', lw=1, alpha=0.4)
ax2.plot([0, shooter_est_pos[0]], [0, shooter_est_pos[1]], 'g:',  lw=1, alpha=0.6)
ax2.annotate('', xy=shooter_est_pos[:2], xytext=SHOOTER_POS[:2],
             arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
ax2.text((SHOOTER_POS[0]+shooter_est_pos[0])/2 + 5,
         (SHOOTER_POS[1]+shooter_est_pos[1])/2,
         f'{err_3D:.1f} m error', color='purple', fontsize=9)
ax2.set_xlabel('X (m)', fontsize=11);  ax2.set_ylabel('Y (m)', fontsize=11)
ax2.set_title(f'3D Localisation Result\n3D position error = {err_3D:.2f} m',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right');  ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pilar_v_localisation.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10.3 — Event Timeline + Complete Summary                               ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
t_bullet_cpa = RANGE / V_BULLET    # approx. time bullet passes closest to array

fig, ax_tl = plt.subplots(figsize=(14, 4.5))
ax_tl.axhline(0, color='gray', lw=1.5, alpha=0.4)

events = [
    (0.0,         'Shot fired\n(t = 0)',           'darkred',   'v'),
    (t_bullet_cpa, f'Bullet passes\n~array plane',  'darkorange','v'),
    (t_ref,        'Shockwave\narrives',            '#2196F3',  '^'),
    (t_mb_ref,     'Muzzle blast\narrives',         '#F44336',  '^'),
]

for t_ev, label, col, mkr in events:
    ax_tl.scatter([t_ev*1e3], [0], color=col, s=220, marker=mkr, zorder=8)
    yoff = 0.18 if mkr == '^' else -0.28
    ax_tl.text(t_ev*1e3, yoff,
               f'{t_ev*1e3:.1f} ms\n{label}',
               ha='center',
               va='bottom' if yoff > 0 else 'top',
               fontsize=9, color=col, fontweight='bold')

ax_tl.annotate('', xy=(t_mb_ref*1e3, 0.6), xytext=(t_ref*1e3, 0.6),
               arrowprops=dict(arrowstyle='<->', color='purple', lw=2.2))
ax_tl.text((t_ref + t_mb_ref)/2*1e3, 0.68,
           f'Dt = {Dt*1e3:.1f} ms  ->  R_hat = {R_exact_est:.0f} m',
           ha='center', fontsize=10, color='purple', fontweight='bold')

ax_tl.set_xlim([-150, t_mb_ref*1e3 + 250])
ax_tl.set_ylim([-0.65, 0.88])
ax_tl.set_xlabel('Time since firing (ms)', fontsize=11)
ax_tl.set_title(f'Sequence of Acoustic Events -- Mach {M} bullet, '
                f'{RANGE:.0f} m range', fontsize=12, fontweight='bold')
ax_tl.set_yticks([]);  ax_tl.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('event_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

W = 68
def row(s): print(f"||  {s:<{W-2}}||")

print("+" + "="*W + "+")
print("||  COMPLETE SIMULATION SUMMARY" + " "*(W-29) + "||")
print("+" + "="*W + "+")
row(f"Mach {M} bullet = {V_BULLET:.0f} m/s  |  {RANGE:.0f} m range  |  "
    f"{Y_MISS:.0f} m miss  |  {L_ARRAY} m array  |  caliber: {CALIBER}")
row(f"fs = {FS//1000} kHz  |  GCC-PHAT x{INTERP}  ->  {dt_eff:.3f} us resolution")

print("+" + "="*W + "+")
row("SHOCKWAVE  (Whitham N-wave)")
row(f"  dP ~ {dPs[0]:.1f} Pa  |  T_N ~ {T_Ns[0]*1e3:.2f} ms  |  "
    f"arrives at t = {t_ref*1e3:.1f} ms")
row(f"  Encodes -> bullet trajectory direction")
row(f"  {'Pair':<7}  {'True tau_SW (us)':>16}  {'Est (us)':>10}  {'Err (us)':>10}")
for i in range(1, 4):
    tt, te, _, _ = tdoa_results[i]
    row(f"  (0,{i})   {tt*1e6:>+16.3f}  {te*1e6:>+10.3f}  {abs(te-tt)*1e6:>10.3f}")

print("+" + "="*W + "+")
row("MUZZLE BLAST  (Friedlander)")
row(f"  dP ~ {dP_mb_all[0]:.2f} Pa  |  t_pos ~ {t_pos_all[0]*1e3:.1f} ms  |  "
    f"arrives at t = {t_mb_ref*1e3:.1f} ms")
row(f"  Encodes -> shooter bearing (azimuth + elevation)")
row(f"  {'Pair':<7}  {'True tau_MB (us)':>16}  {'Est (us)':>10}  {'Err (us)':>10}")
for i in range(1, 4):
    tt, te, _, _ = tdoa_mb_results[i]
    row(f"  (0,{i})   {tt*1e6:>+16.3f}  {te*1e6:>+10.3f}  {abs(te-tt)*1e6:>10.3f}")

print("+" + "="*W + "+")
row("BULLET SPEED (Section 9)")
row(f"  Method A (angle only)        : Mach {M_A:.3f} = {M_A*C:.1f} m/s")
row(f"  Method B (duration+amplitude): Mach {M_B:.3f} = {M_B*C:.1f} m/s")
row(f"  Method C (geometric corr.)   : Mach {M_C:.3f} = {M_C*C:.1f} m/s")

print("+" + "="*W + "+")
row("PILAR-V 3D LOCALISATION")
row(f"  Dt = {Dt*1e3:.3f} ms  |  dR/dDt ~ {dR_dDt:.0f} m/s")
row(f"  DOA (from MB)  : AZ = {az_est:.2f} deg   EL = {el_est:.2f} deg   "
    f"(errors: {az_err:.3f} deg, {el_err:.3f} deg)")
row(f"  Range (PILAR-V): {R_exact_est:.1f} m   "
    f"(error vs along-track: {R_exact_est-R_along_actual:+.1f} m)")
row(f"  3D estimate    : [{shooter_est_pos[0]:+.1f}, "
    f"{shooter_est_pos[1]:+.1f}, {shooter_est_pos[2]:+.1f}] m")
row(f"  Actual position: [{SHOOTER_POS[0]:+.1f}, "
    f"{SHOOTER_POS[1]:+.1f}, {SHOOTER_POS[2]:+.1f}] m")
row(f"  3D error       : {err_3D:.2f} m")
print("+" + "="*W + "+")

