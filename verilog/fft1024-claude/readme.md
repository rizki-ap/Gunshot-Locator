# 1024-Point FFT/IFFT — Verilog Implementation
## For GCC-PHAT on Cyclone V FPGA

---

## File Structure

| File | Description |
|---|---|
| `fft1024.v` | Top-level FFT/IFFT module (10-stage radix-2 DIT) |
| `fft_butterfly.v` | Pipelined radix-2 butterfly unit (3-cycle latency) |
| `fft_bitrev.v` | Bit-reversal permutation module |
| `fft_twiddle_rom.v` | 512-entry twiddle factor ROM (cos/sin) |
| `tb_fft1024.v` | Testbench with 3 test cases |
| `gen_twiddle.py` | Python script to generate twiddle hex files |
| `Makefile` | Build and simulation flow |

---

## Quick Start

### 1. Generate Twiddle Files
```bash
python3 gen_twiddle.py
# Creates: twiddle_cos.hex, twiddle_sin.hex
```

### 2. Simulate (Icarus Verilog)
```bash
make all
# Or step by step:
make twiddle
make compile
make sim
make wave     # Opens GTKWave
```

### 3. View Waveform
```bash
gtkwave tb_fft1024.vcd
```

---

## Architecture

```
Input samples (N=1024)
       |
  [Bit-Reversal]        — reorders input for DIT FFT
       |
  [Stage 0]             — 512 butterflies, span=1
  [Stage 1]             — 512 butterflies, span=2
  ...
  [Stage 9]             — 512 butterflies, span=512
       |
  Output spectrum X[k]  — N complex values
```

### Fixed-Point Format
- **18-bit signed, Q1.17**
- Range: -1.0 to +0.9999924 (≈ ±131071)
- Twiddle factors stored as Q1.17 in ROM

### IFFT Mode
- Set `inverse = 1`
- Twiddle factors are conjugated (sine negated) automatically
- Output is NOT scaled by 1/N — divide manually or shift right by LOG2N=10

---

## Port Description — fft1024

| Port | Dir | Width | Description |
|---|---|---|---|
| `clk` | in | 1 | System clock |
| `rst_n` | in | 1 | Active-low async reset |
| `inverse` | in | 1 | 0=FFT, 1=IFFT |
| `start` | in | 1 | Pulse high to begin |
| `in_re` | in | 18 | Input real part |
| `in_im` | in | 18 | Input imaginary part |
| `in_valid` | in | 1 | Qualify input samples |
| `out_re` | out | 18 | Output real part |
| `out_im` | out | 18 | Output imaginary part |
| `out_valid` | out | 1 | Qualifies output |
| `done` | out | 1 | Single-cycle pulse when done |

---

## Timing

| Parameter | Value |
|---|---|
| Clock | 50 MHz (Cyclone V max ~150 MHz for this design) |
| Butterfly latency | 3 cycles |
| Bit-reversal latency | 2×N = 2048 cycles |
| Compute latency | 10 × 512 = 5120 cycles |
| Total per transform | ~7200 cycles |
| Throughput at 50 MHz | ~6944 transforms/sec |

---

## Quartus Integration (Cyclone V)

1. Add all `.v` files to your Quartus project
2. Set `fft1024` as a sub-module in your GCC-PHAT top-level
3. Initialize ROM via `$readmemh` — Quartus infers M10K block RAM
4. Instantiate two copies (one per microphone channel)
5. Use Platform Designer (Qsys) to connect HPS AXI bridge

### Estimated Resource Usage (Cyclone V 5CSXFC6D6F31)
| Resource | Used | Available | % |
|---|---|---|---|
| ALMs | ~3,800 | 41,910 | 9% |
| DSP blocks | ~36 | 112 | 32% |
| M10K blocks | ~23 | 553 | 4% |

---

## Test Cases

| Test | Input | Expected Result |
|---|---|---|
| 1 | cos(2π·8·n/N) | Peak at bin 8 |
| 2 | DC = 0.5 | Magnitude at bin 0 ≈ 0.5, others ≈ 0 |
| 3 | 3-tone mix → FFT → IFFT | Reconstruction error < 0.01 |

---

## Known Limitations

1. **Scaling**: Each butterfly stage divides by 2 to prevent overflow. After 10 stages, output is scaled by 1/1024 relative to full-scale. Account for this in downstream GCC-PHAT computation.
2. **IFFT 1/N**: Not applied internally. Shift right by 10 bits or absorb into cross-spectrum normalization.
3. **Overflow**: Input amplitudes should be < 0.5 to avoid saturation through all 10 stages.

---

## Next Steps for GCC-PHAT Paper

1. Instantiate two `fft1024` modules (mic1, mic2)
2. Add `gcc_phat_core.v` for cross-spectrum + PHAT normalization
3. Add IFFT for correlation
4. Add peak detector for TDOA output
5. Connect to HPS via AXI lightweight bridge
