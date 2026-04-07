#!/usr/bin/env python3
"""
gen_recip_rom.py
Generate reciprocal lookup table for PHAT normalization.

The ROM stores: recip[i] = round(2^17 / (i+1)) for i=0..1023
where index i = top 10 bits of the magnitude |G[k]|.

This avides a hardware divider — the PHAT normalization
G_phat = G / |G|  becomes  G_phat = G * recip[|G|>>8]

Format: 18-bit unsigned Q0.17, two's complement hex
"""

ENTRIES    = 1024       # 2^LOG2N
FRAC_BITS  = 17         # Q0.17 reciprocal
SCALE      = 2**FRAC_BITS   # 131072
BITS       = 18
MASK       = (1 << BITS) - 1

lines = []
for i in range(ENTRIES):
    # Map index i to magnitude value
    # Index 0 = magnitude near zero -> clamp reciprocal to max
    if i == 0:
        val = MASK  # maximum (1.0 in Q0.17 terms)
    else:
        # The index represents the top 10 bits of an 18-bit magnitude
        # Actual magnitude midpoint = (i + 0.5) * (2^(18-10)) = (i+0.5)*256
        mag = (i + 0.5) * (2 ** (BITS - 10))
        recip = SCALE / mag
        val = int(round(recip))
        val = min(val, MASK)  # clamp to 18-bit max

    lines.append(format(val & MASK, '05X'))

with open('recip_rom.hex', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Generated recip_rom.hex ({ENTRIES} entries)")
print(f"Format: 18-bit unsigned Q0.17 hex")
print(f"Index 0   (mag~0):   {lines[0]} = {int(lines[0],16)}")
print(f"Index 1   (mag~384): {lines[1]} = {int(lines[1],16)} -> {int(lines[1],16)/SCALE:.5f}")
print(f"Index 128 (mag~32k): {lines[128]} = {int(lines[128],16)} -> {int(lines[128],16)/SCALE:.5f}")
print(f"Index 512 (mag~128k):{lines[512]} = {int(lines[512],16)} -> {int(lines[512],16)/SCALE:.5f}")
