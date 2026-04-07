# ============================================================
#  Makefile — GCC-PHAT FFT/IFFT Simulation
#  Tools: Icarus Verilog (iverilog) + GTKWave
# ============================================================

SRCS_FFT = fft_twiddle_rom.v \
           fft_butterfly.v   \
           fft_bitrev.v      \
           fft1024.v         \
           tb_fft1024.v

SRCS_RFFT = fft_twiddle_rom.v   \
            fft_butterfly.v     \
            fft_bitrev.v        \
            fft1024.v           \
            rfft1024_packed.v   \
            tb_rfft1024_packed.v

.PHONY: all twiddle compile_fft compile_rfft sim_fft sim_rfft wave_fft wave_rfft clean

all: twiddle compile_fft sim_fft compile_rfft sim_rfft

twiddle:
	python3 gen_twiddle.py

compile_fft:
	iverilog -o sim_fft -Wall $(SRCS_FFT)

sim_fft: compile_fft
	vvp sim_fft

wave_fft:
	gtkwave tb_fft1024.vcd &

compile_rfft:
	iverilog -o sim_rfft -Wall $(SRCS_RFFT)

sim_rfft: compile_rfft
	vvp sim_rfft

wave_rfft:
	gtkwave tb_rfft1024_packed.vcd &

clean:
	rm -f sim_fft sim_rfft *.vcd twiddle_cos.hex twiddle_sin.hex

# ============================================================
# Full GCC-PHAT pipeline simulation
# ============================================================
SRCS_FULL = fft_twiddle_rom.v   \
            fft_butterfly.v     \
            fft_bitrev.v        \
            fft1024.v           \
            rfft1024_packed.v   \
            gcc_phat_core.v     \
            gcc_phat_top.v      \
            peak_detector_tdoa.v \
            tb_gcc_phat_top.v

compile_full:
	iverilog -o sim_full -Wall $(SRCS_FULL)

sim_full: compile_full
	vvp sim_full

wave_full:
	gtkwave tb_gcc_phat_top.vcd &

all_full: twiddle recip compile_full sim_full

recip:
	python3 gen_recip_rom.py
