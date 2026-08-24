# ╔═════════════════════════════════════════════╗
# ║  SECTION 1.1 — Imports & Physical Constants ║
# ╚═════════════════════════════════════════════╝
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Physical constants ─────────────────────────────────────────────────────────
C     = 343.0      # Speed of sound (m/s) -- ISA sea level
P_ATM = 101325.0   # Atmospheric pressure (Pa)

MIC_SEPARATION = 0.30        # Tetrahedral edge length (m) -- ASSUMED SENSOR GEOMETRY
ADC_FS         = 100_000   # Sample rate (Hz) -- ASSUMED ADC SETTING
ADC_BIT_DEPTH  = 16         # ADC quantization depth -- ASSUMED ADC SETTING
MIC_LO_FREQ, MIC_HI_FREQ = 40.0, 20000.0   # mic passband (Hz) -- ASSUMED SENSOR RESPONSE
M        = 2.5      # Mach number -- ASSUMED BULLET VELOCITY
BULLET_SPEED = M * C     # Bullet speed (m/s)


BULLET_LIBRARY = {
    '7.62_NATO': dict(L=0.028, dP0_sw=7.5, b0_sw=50.0,
                       P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003),
    '5.56_NATO': dict(L=0.023, dP0_sw=7.5, b0_sw=50.0,   # dP0_sw not independently re-fit
                       P_REF_MB=200.0, R_REF_MB=10.0, T_POS_REF=0.003/4.0),
}
CALIBER = '5.56_NATO'   # Ruger .223/5.56 -- switch to '7.62_NATO' for the NATO reference
bl = BULLET_LIBRARY[CALIBER]

BULLET_RANGE  = 200.0   # Along-track shooter range (m)      -- ASSUMED TRAJECTORY
BULLET_MISS = 10.0      # Bullet clears array by this much (m) -- ASSUMED TRAJECTORY

BULLET_ORIGIN = np.array([-BULLET_RANGE, BULLET_MISS, 0.0])   # bullet position at t=0 (also muzzle position)
BULLET_TRAJ   = np.array([ 1.0,   0.0,   0.0]) # bullet velocity unit vector
BULLET_STOP   = np.array([ 0.0,   0.0,   0.0]) # bullet stop position
V = BULLET_STOP - BULLET_ORIGIN
V_HAT = V / np.linalg.norm(V)

PRE_ROLL  = 0.010         # 10 ms before the first shockwave arrival
POST_ROLL = 0.150         # 150 ms after the last muzzle-blast arrival (covers Friedlander decay)

NOISE_FLOOR_PA = 0.005    # simple model: continuous sensor self-noise (~48 dB SPL)

RT60         = 1.25    # room decay time constant (s) -- raise for "boomier" indoor spaces
NOISE_RMS_PA = 0.075   # background noise floor in Pa -- controls peak/noise ratio
NOISE_SLOPE  = -2.4    # PSD slope: 0=white, -1=pink, -2=brown

REAL_NOISE_RMS   = 0.000595   # measured background RMS (normalized units)
REAL_RT60        = 1.32       # measured reverb decay time (s)
REAL_PEAK_NORM   = 0.098      # measured peak amplitude (normalized units)
REAL_NOISE_SLOPE = -2.46      # measured PSD slope (brown-ish noise)

# ╔══════════════════════════════════════════════════════╗
# ║  SECTION 1.2 — Tetrahedral Microphone Array Geometry ║
# ╚══════════════════════════════════════════════════════╝
#  Regular tetrahedron from alternating +-1 cube-vertex coordinates:
#
#     (1,1,1)  (1,-1,-1)  (-1,1,-1)  (-1,-1,1)
#
#  All 6 edges of this subset have identical length 2*sqrt(2).
#  Scale by  L / (2*sqrt(2))  to hit the target aperture.

COLORS  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

def make_tetrahedron(edge_length):
    raw = np.array([[0.000, 0.000, 1.000], [0.000, 0.943, -0.333],
                     [-0.816, -0.471, -0.333], [0.816, -0.471, -0.333]], dtype=float)
    scale = edge_length / np.linalg.norm(raw[0] - raw[1])   # raw edge = 2*sqrt(2)
    return raw * scale

mic_pos = make_tetrahedron(MIC_SEPARATION)

print("---------------Microphone Setting----------------------------------")
print("Microphone positions (m):")
for i, p in enumerate(mic_pos):
    print(f"  M{i}: [{p[0]:+.5f}, {p[1]:+.5f}, {p[2]:+.5f}]")
print("Edge-length verification:")
for i in range(4):
    for j in range(i+1, 4):
        d = np.linalg.norm(mic_pos[i] - mic_pos[j])
        print(f"  M{i}-M{j}: {d:.6f} m  {'OK' if abs(d-MIC_SEPARATION)<1e-3 else 'MISMATCH'}")

# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 2.1 — Sample Rate, Bit Depth & Mic Frequency Response ║
# ╚════════════════════════════════════════════════════════════════╝
from scipy.signal import butter, sosfiltfilt, fftconvolve

def mic_bandpass(x, fs, lo=MIC_LO_FREQ, hi=MIC_HI_FREQ, order=4):
    """Simulate microphone/ADC anti-alias frequency response."""
    sos = butter(order, [lo, min(hi, fs/2*0.99)], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

def quantize(x, bits=ADC_BIT_DEPTH, full_scale=1.0):
    """Simulate ADC quantization."""
    levels = 2 ** (bits - 1)
    return np.round(np.clip(x, -full_scale, full_scale) * levels) / levels

print("----------------ADC & Noise Settings------------------------")
print(f"ADC settings: fs={ADC_FS/1000:.0f} kHz, {ADC_BIT_DEPTH}-bit, "
      f"Microphone passband [{MIC_LO_FREQ:.0f}, {MIC_HI_FREQ:.0f}] Hz")

# ╔═════════════════════════════════════════╗
# ║  SECTION 3.1 — Bullet / Caliber Library ║
# ╚═════════════════════════════════════════╝
#  Friedlander & Whitham scaling constants are caliber-specific. The 5.56 entry
#  below is a rough fit derived from only 2 real Ruger recordings -- treat it
#  as a starting point, not a fully validated reference. Switch CALIBER to
#  '7.62_NATO' to fall back to the original long-range reference values.

L_BULLET  = bl['L']           # Projectile length (m) -- scales N-wave duration
dP0_sw    = bl['dP0_sw']      # Reference shockwave peak overpressure (Pa)
b0_sw     = bl['b0_sw']       # Reference miss distance for dP0_sw (m)
P_REF_MB  = bl['P_REF_MB']    # Reference muzzle blast overpressure (Pa)
R_REF_MB  = bl['R_REF_MB']    # Reference range for P_REF_MB (m)
T_POS_REF = bl['T_POS_REF']   # Reference Friedlander positive-phase duration (s)

print("---------------------Bullet Setting-----------------------")
print(f"Bullet property: {CALIBER}")
print(f"- L_BULLET = {L_BULLET*1000:.1f} mm")
print(f"- P_REF_MB = {P_REF_MB:.0f} Pa @ {R_REF_MB:.0f} m, T_POS_REF = {T_POS_REF*1e3:.2f} ms")


# ╔═════════════════════════════════════════════════════╗
# ║  SECTION 3.2 — Assumed Bullet Velocity (Mach Number)║
# ╚═════════════════════════════════════════════════════╝

# Mach cone geometry
MU   = np.arcsin(1.0 / M)        # Mach half-angle  mu = arcsin(1/M)
BETA = np.sqrt(M**2 - 1.0)       # beta = sqrt(M^2-1)

print(f"Bullet         : Mach {M} = {BULLET_SPEED:.0f} m/s")
print(f"Mach angle mu  : {np.degrees(MU):.2f} deg -> Mach cone angle")
print(f"beta = sqrt(M^2-1): {BETA:.4f}  -> Mach wave parameter, tangent of Mach angle")
print(f"Array aperture : {MIC_SEPARATION}m -> max TDOA = {MIC_SEPARATION/C*1e6:.1f}us "
      f"({MIC_SEPARATION/C*ADC_FS:.2f}sample)")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4.1 — Shooter Origin, Trajectory Direction & Miss Distance║
# ╚════════════════════════════════════════════════════════════════════╝
#  Shooter at (-BULLET_RANGE, BULLET_MISS, 0).  Array centroid at origin.
#  Bullet flies in +x direction with BULLET_MISS lateral miss distance.
#
#   Y
#   |        * shooter (-BULLET_RANGE, BULLET_MISS)
#   |          -----------------------------> bullet velocity
#   |
#   0------------------------------------ X
#  array (0,0,0)


# The muzzle blast originates from the same point the bullet trajectory starts:
# the shooter IS the bullet's origin.
SHOOTER_POS = BULLET_ORIGIN.copy()

print("---------------------Bullet Setting-----------------------")
print(f"Shooter / bullet origin : [{SHOOTER_POS[0]:.1f}, {SHOOTER_POS[1]:.1f}, {SHOOTER_POS[2]:.1f}] m")
print(f"Trajectory direction    : {BULLET_TRAJ}")
print(f"Range (along-track)     : {BULLET_RANGE:.1f} m   |   Miss distance: {BULLET_MISS:.1f} m")


# ╔══════════════════════════════════════════╗
# ║  SECTION 5.1 — Shockwave Arrival Physics ║
# ╚══════════════════════════════════════════╝
#  DERIVATION
#  ----------
#  Bullet position at emission time t_e:  p(t_e) = p0 + BULLET_SPEED . t_e . v_hat
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
            "Increase BULLET_RANGE or reduce BULLET_MISS."
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
    ta, te, a, b = shockwave_arrival(p, BULLET_ORIGIN, BULLET_TRAJ, M, C)
    t_arr[i]=ta; t_emi[i]=te; a_all[i]=a; b_all[i]=b
    print(f"  M{i}   {b:>10.5f}  {a:>12.4f}  {te*1e3:>14.6f}  {ta*1e3:>15.6f}")

t_ref, _, a_ref, b_ref = shockwave_arrival(np.zeros(3), BULLET_ORIGIN, BULLET_TRAJ, M, C)
print(f"\n  Centroid:  b={b_ref:.4f} m  |  t_ref={t_ref*1e3:.6f} ms")

bullet_x_at_emit = -BULLET_RANGE + BULLET_SPEED * t_emi.mean()
bullet_x_at_arrv = -BULLET_RANGE + BULLET_SPEED * t_arr.mean()
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

ax.fill_between(x_traj, BULLET_MISS-cone_half, BULLET_MISS+cone_half,
                alpha=0.12, color='darkorange')
ax.plot(x_traj, BULLET_MISS+cone_half, '--', color='darkorange', lw=1.0, alpha=0.7,
        label='Mach cone')
ax.plot(x_traj, BULLET_MISS-cone_half, '--', color='darkorange', lw=1.0, alpha=0.7)

ax.annotate('', xy=(250, BULLET_MISS), xytext=BULLET_ORIGIN[:2],
            arrowprops=dict(arrowstyle='->', color='crimson', lw=2.2))
ax.text(-700, BULLET_MISS+4, f'Mach {M}  ->', color='crimson',
        fontsize=10, fontweight='bold')
ax.scatter(*BULLET_ORIGIN[:2], color='darkred', s=250, zorder=8,
           marker='*', label=f'Shooter ({BULLET_RANGE:.0f} m)')
ax.scatter([0], [0], color='black', s=120, zorder=8, marker='^',
           label='Array centroid')

ax.annotate('', xy=(0, BULLET_MISS), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(8, BULLET_MISS/2, f'{BULLET_MISS:.0f} m\nmiss', color='gray',
        fontsize=9, va='center')
ax.plot([0,0], [0, BULLET_MISS], ':', color='gray', lw=1.0)

emission_x = -BULLET_RANGE + BULLET_SPEED * t_emi.mean()
r_wave     = b_all.mean() * M / BETA
theta_arc  = np.linspace(-np.pi*0.55, np.pi*0.55, 80)
ax.plot(emission_x + r_wave*np.cos(theta_arc),
        BULLET_MISS    + r_wave*np.sin(theta_arc),
        color='steelblue', lw=1.6, alpha=0.8, label='Wavefront arc')

for p, col in zip(mic_pos, COLORS):
    ax.scatter(p[0], p[1], color=col, s=70, zorder=9)

ax.set_xlim([-1060, 300]);  ax.set_ylim([-10, 130])
ax.set_xlabel('X (m)', fontsize=11); ax.set_ylabel('Y (m)', fontsize=11)
ax.set_title(f'Scene Overview -- Top View\nMach {M} bullet, {BULLET_RANGE:.0f} m range, {BULLET_MISS:.0f} m miss',
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

dx = 0 - emission_x;  dy = 0 - BULLET_MISS
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


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5.3 — N-wave (Shockwave) Signal Generation (Whitham weak-shock model)║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
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
t_sig   = np.arange(t_start, t_end, 1.0 / ADC_FS)

rng     = np.random.default_rng(seed=42)
signals = [];  T_Ns = [];  dPs = []

print(f"N-wave parameters (SNR = 20 dB, fs = {ADC_FS//1000} kHz)")
print(f"{'Mic':<5} {'b (m)':>10} {'T_N (ms)':>12} {'dP (Pa)':>10}")
for i in range(4):
    sig, T_N, dP = generate_nwave(
        t_sig, t_arr[i], b_all[i], M, C, L_BULLET, snr_db=20, rng=rng)
    signals.append(sig);  T_Ns.append(T_N);  dPs.append(dP)
    print(f"  M{i}   {b_all[i]:>10.4f}  {T_N*1e3:>12.4f}  {dP:>10.3f}")


# ╔════════════════════════════════════╗
# ║  SECTION 5.4 — N-wave Signal Plots ║
# ╚════════════════════════════════════╝
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
    f'N-wave Shockwave Signals -- Mach {M}, {BULLET_RANGE:.0f} m range, '
    f'{BULLET_MISS:.0f} m miss distance\n'
    f'T_N ~ {T_Ns[0]*1e3:.2f} ms  |  dP ~ {dPs[0]:.1f} Pa  |  '
    f'SNR = 20 dB  |  fs = {ADC_FS//1000} kHz',
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
#   Dt         ->  shooter BULLET_RANGE                (PILAR-V formula, Section 10)
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
                      1.0 / ADC_FS)

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
    f'Friedlander Muzzle Blast Signals -- {BULLET_RANGE:.0f} m range\n'
    f'dP ~ {dP_mb_all[0]:.2f} Pa  |  t_pos ~ {t_pos_all[0]*1e3:.1f} ms  '
    f'(dotted line)  |  SNR = 20 dB',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('muzzle_blast_signals.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔═════════════════════════════════════════════════════════════╗
# ║  SECTION 5.8 — Combined Per-Sensor Signal (ideal, noiseless)║
# ╚═════════════════════════════════════════════════════════════╝
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


t0 = min(t_arr.min(), t_mb.min()) - PRE_ROLL
t1 = t_mb.max() + POST_ROLL
t_master = np.arange(t0, t1, 1.0 / ADC_FS)

print(f"Master ADC timeline : {t0:.4f} s  ->  {t1:.4f} s")
print(f"Samples             : {len(t_master)}  ({len(t_master)/ADC_FS:.3f} s @ {ADC_FS/1000:.0f} kHz)")

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


# ╔═══════════════════════════════════════════════════════════════╗
# ║  SECTION 7.2 — Combined Signal Plots (flat-noise-floor model) ║
# ╚═══════════════════════════════════════════════════════════════╝
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
    f'Combined per-sensor received signal -- Mach {M}, {BULLET_RANGE:.0f} m range\n'
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
    rir = make_room_ir(RT60, ADC_FS, seed=1000+i)
    wet = fftconvolve(dry, rir, mode='full')[:len(dry)]
    if np.abs(dry).max() > 0:
        wet *= np.abs(dry).max() / (np.abs(wet).max() + 1e-12)   # preserve peak

    # 2) Colored background noise
    noise = make_colored_noise(len(t_master), ADC_FS, slope=NOISE_SLOPE,
                                rms=NOISE_RMS_PA, seed=2000+i)

    # 3) Mic/ADC bandpass response (Section 2.1)
    combined = mic_bandpass(wet + noise, ADC_FS, lo=MIC_LO_FREQ, hi=MIC_HI_FREQ)
    realistic_signals_pa.append(combined)

# -- Calibrate Pa -> normalized ADC units by PEAK match (prevents clipping) ---
peak_pa = max(np.abs(s).max() for s in realistic_signals_pa)
PA_TO_NORM = REAL_PEAK_NORM / peak_pa

realistic_signals = [quantize(s * PA_TO_NORM, bits=ADC_BIT_DEPTH) for s in realistic_signals_pa]

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
tail = realistic_signals[0][mb_idx:mb_idx+int(1.0*ADC_FS)]
win_e = int(0.005*ADC_FS)
t_pts, e_pts = [], []
for k in range(0, len(tail)-win_e, win_e):
    r = np.sqrt(np.mean(tail[k:k+win_e]**2))
    if r > 3*noise_rms:
        t_pts.append(k/ADC_FS); e_pts.append(20*np.log10(r))
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
