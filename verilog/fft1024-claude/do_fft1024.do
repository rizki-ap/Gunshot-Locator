# Create working library
vlib work
vmap work work

# Compile design files
vlog fft1024.v
vlog fft_butterfly.v
vlog fft_bitrev.v
vlog fft_twiddle_rom.v

# Compile testbench
vlog tb_fft1024.v

# Start simulation
vsim -voptargs=+acc work.tb_fft1024

# Add waveforms
add wave -divider "Clock & Reset"
add wave sim:/tb_fft1024/clk
add wave sim:/tb_fft1024/rst_n

add wave -divider "Input Signals"
#add wave sim:/tb_fft1024/in_re
#add wave sim:/tb_fft1024/in_im
add wave sim:/tb_fft1024/in_valid

add wave -divider "Output Signals"
#add wave sim:/tb_fft1024/out_re
#add wave sim:/tb_fft1024/out_im
add wave sim:/tb_fft1024/out_valid

# Optional: internal FFT signals (debug)
add wave -divider "Internal"
#add wave sim:/tb_fft1024/uut/*

# Run simulation
run 10 us

# Zoom waveform
wave zoom full