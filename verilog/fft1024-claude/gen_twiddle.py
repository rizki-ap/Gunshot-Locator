#!/usr/bin/env python3
"""
gen_twiddle.py
Generate twiddle factor ROM hex files for 1024-point FFT.

Outputs:
  twiddle_cos.hex  - cosine values  (W_N^k real part)
  twiddle_sin.hex  - -sine values   (W_N^k imag part, negative for DIT FFT)

Fixed-point format: 18-bit signed, Q1.17
  +1.0 -> 0x1FFFF (131071)
  -1.0 -> 0x20000 (-131072, two's complement)
"""

import math

NFFT       = 1024
HALF       = NFFT // 2     # 512 entries needed
FRAC_BITS  = 17            # Q1.17 format
SCALE      = 2 ** FRAC_BITS  # 131072
BITS       = 18
MASK       = (1 << BITS) - 1

def to_twos_complement(val, bits):
    """Convert signed integer to two's complement hex string."""
    if val < 0:
        val = val + (1 << bits)
    return format(val & MASK, f'05X')  # 5 hex digits for 18 bits

cos_lines = []
sin_lines = []

for k in range(HALF):
    angle   = 2.0 * math.pi * k / NFFT
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)   # note: twiddle W = cos - j*sin

    # Quantise to Q1.17
    cos_q = int(round(cos_val * SCALE))
    sin_q = int(round(sin_val * SCALE))

    # Clamp to 18-bit signed range
    cos_q = max(-(1 << (BITS-1)), min((1 << (BITS-1))-1, cos_q))
    sin_q = max(-(1 << (BITS-1)), min((1 << (BITS-1))-1, sin_q))

    cos_lines.append(to_twos_complement(cos_q, BITS))
    sin_lines.append(to_twos_complement(sin_q, BITS))   # positive sine stored; module negates for IFFT

with open('twiddle_cos.hex', 'w') as f:
    f.write('\n'.join(cos_lines) + '\n')

with open('twiddle_sin.hex', 'w') as f:
    f.write('\n'.join(sin_lines) + '\n')

print(f"Generated twiddle_cos.hex and twiddle_sin.hex ({HALF} entries each)")
print(f"Format: 18-bit signed Q1.17, two's complement hex")
print(f"k=0:   cos={cos_lines[0]}, sin={sin_lines[0]}")
print(f"k=128: cos={cos_lines[128]}, sin={sin_lines[128]}")
print(f"k=256: cos={cos_lines[256]}, sin={sin_lines[256]}")
