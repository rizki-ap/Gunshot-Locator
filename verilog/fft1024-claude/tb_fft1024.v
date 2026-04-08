// ============================================================
//  tb_fft1024.v
//  Testbench for fft1024.v
//
//  Tests:
//    1. Single-tone sine wave -> verify spectral peak at correct bin
//    2. Forward FFT then IFFT -> verify reconstruction
//    3. DC input -> verify bin 0 only
//
//  Dumps waveform to tb_fft1024.vcd for GTKWave viewing
// ============================================================
`timescale 1ns/1ps

module tb_fft1024;

    // -------------------------------------------------------
    // Parameters
    // -------------------------------------------------------
    localparam DATA_WIDTH = 18;
    localparam NFFT       = 1024;
    localparam LOG2N      = 10;
    localparam CLK_PERIOD = 20;  // 50 MHz clock (20 ns)
    localparam FRAC       = 17;  // Q1.17 fractional bits
    localparam SCALE      = 2**FRAC; // 131072

    // -------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------
    reg        clk, rst_n, inverse, start, in_valid;
    reg  signed [DATA_WIDTH-1:0] in_re, in_im;
    wire signed [DATA_WIDTH-1:0] out_re, out_im;
    wire       out_valid, done;

    // -------------------------------------------------------
    // Instantiate FFT
    // -------------------------------------------------------
    fft1024 #(
        .DATA_WIDTH    (DATA_WIDTH),
        .TWIDDLE_WIDTH (DATA_WIDTH),
        .NFFT          (NFFT),
        .LOG2N         (LOG2N)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .inverse   (inverse),
        .start     (start),
        .in_re     (in_re),
        .in_im     (in_im),
        .in_valid  (in_valid),
        .out_re    (out_re),
        .out_im    (out_im),
        .out_valid (out_valid),
        .done      (done)
    );

    // -------------------------------------------------------
    // Clock generation
    // -------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------
    // Storage for outputs
    // -------------------------------------------------------
    real fft_re [0:NFFT-1];
    real fft_im [0:NFFT-1];
    real ifft_re [0:NFFT-1];
    real ifft_im [0:NFFT-1];
    integer out_idx;
    integer max_bin, i;
    real max_mag, mag;

    // -------------------------------------------------------
    // Waveform dump
    // -------------------------------------------------------
    initial begin
        $dumpfile("tb_fft1024.vcd");
        $dumpvars(0, tb_fft1024);
    end

    // -------------------------------------------------------
    // Task: apply N samples from arrays
    // -------------------------------------------------------
    real input_re [0:NFFT-1];
    real input_im [0:NFFT-1];

    task apply_input;
        integer n;
        begin
            @(posedge clk); #1;
            start    = 1;
            in_valid = 0;
            @(posedge clk); #1;
            start = 0;

            for (n = 0; n < NFFT; n = n+1) begin
                in_valid = 1;
                in_re    = $rtoi(input_re[n] * SCALE);
                in_im    = $rtoi(input_im[n] * SCALE);
                @(posedge clk); #1;
            end
            in_valid = 0;
        end
    endtask

    // -------------------------------------------------------
    // Task: collect N output samples
    // -------------------------------------------------------
    task collect_output_fft;
        begin
            out_idx = 0;
            @(posedge done);
            // Wait and capture on out_valid pulses before done
        end
    endtask

    // -------------------------------------------------------
    // Output capture process
    // -------------------------------------------------------
    reg capture_fft_flag, capture_ifft_flag;
    integer fft_out_cnt, ifft_out_cnt;

    initial begin
        capture_fft_flag  = 0;
        capture_ifft_flag = 0;
        fft_out_cnt       = 0;
        ifft_out_cnt      = 0;
    end

    always @(posedge clk) begin
        if (out_valid) begin
            if (capture_fft_flag && fft_out_cnt < NFFT) begin
                fft_re[fft_out_cnt] = $itor(out_re) / SCALE;
                fft_im[fft_out_cnt] = $itor(out_im) / SCALE;
                fft_out_cnt = fft_out_cnt + 1;
            end
            if (capture_ifft_flag && ifft_out_cnt < NFFT) begin
                ifft_re[ifft_out_cnt] = $itor(out_re) / SCALE;
                ifft_im[ifft_out_cnt] = $itor(out_im) / SCALE;
                ifft_out_cnt = ifft_out_cnt + 1;
            end
        end
    end

    // -------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------
    initial begin
        // Initialise
        rst_n    = 0;
        start    = 0;
        inverse  = 0;
        in_valid = 0;
        in_re    = 0;
        in_im    = 0;
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(4) @(posedge clk);

        // ==================================================
        // TEST 1: Single tone, bin 8
        //   x[n] = cos(2*pi*8*n/N)
        //   Expected: FFT peak at bins 8 and N-8 (1016)
        // ==================================================
        $display("\n=== TEST 1: Single-tone FFT (bin 8) ===");
        for (i = 0; i < NFFT; i = i+1) begin
            input_re[i] = $cos(2.0 * 3.14159265358979 * 8.0 * i / NFFT);
            input_im[i] = 0.0;
        end

        inverse           = 0;
        capture_fft_flag  = 1;
        fft_out_cnt       = 0;

        apply_input();
        @(posedge done);
        capture_fft_flag = 0;

        // Find peak bin
        max_mag = 0.0;
        max_bin = 0;
        for (i = 0; i < NFFT; i = i+1) begin
            mag = fft_re[i]*fft_re[i] + fft_im[i]*fft_im[i];
            if (mag > max_mag) begin
                max_mag = mag;
                max_bin = i;
            end
        end
        $display("  Peak bin: %0d (expected 8)", max_bin);
        $display("  Magnitude at bin 8: %.4f", $sqrt(fft_re[8]*fft_re[8] + fft_im[8]*fft_im[8]));
        $display("  Magnitude at bin 0: %.4f", $sqrt(fft_re[0]*fft_re[0] + fft_im[0]*fft_im[0]));
        if (max_bin == 8)
            $display("  TEST 1 PASSED");
        else
            $display("  TEST 1 FAILED");

        repeat(10) @(posedge clk);

        // ==================================================
        // TEST 2: DC input
        //   x[n] = 1.0
        //   Expected: FFT magnitude at bin 0 = N, all others ~0
        // ==================================================
        $display("\n=== TEST 2: DC Input ===");
        for (i = 0; i < NFFT; i = i+1) begin
            input_re[i] = 0.5;  // scaled to avoid overflow after N additions
            input_im[i] = 0.0;
        end

        inverse          = 0;
        capture_fft_flag = 1;
        fft_out_cnt      = 0;

        apply_input();
        @(posedge done);
        capture_fft_flag = 0;

        $display("  Magnitude bin 0: %.4f (expected ~%.4f)",
            $sqrt(fft_re[0]*fft_re[0] + fft_im[0]*fft_im[0]),
            0.5);
        $display("  Magnitude bin 1: %.6f (expected ~0.0)",
            $sqrt(fft_re[1]*fft_re[1] + fft_im[1]*fft_im[1]));

        repeat(10) @(posedge clk);

        // ==================================================
        // TEST 3: FFT then IFFT round-trip
        //   Use a random-looking bandlimited signal
        //   After FFT->IFFT, output should match input
        // ==================================================
        $display("\n=== TEST 3: FFT -> IFFT Round-Trip ===");
        for (i = 0; i < NFFT; i = i+1) begin
            // Pseudo-random bandlimited: sum of 3 tones
            input_re[i] = 0.3 * $cos(2.0*3.14159265*3.0*i/NFFT)
                        + 0.5 * $cos(2.0*3.14159265*17.0*i/NFFT)
                        + 0.2 * $cos(2.0*3.14159265*50.0*i/NFFT);
            input_im[i] = 0.0;
        end

        // Forward FFT
        inverse          = 0;
        capture_fft_flag = 1;
        fft_out_cnt      = 0;
        apply_input();
        @(posedge done);
        capture_fft_flag = 0;

        repeat(10) @(posedge clk);

        // Feed FFT output into IFFT
        for (i = 0; i < NFFT; i = i+1) begin
            input_re[i] = fft_re[i];
            input_im[i] = fft_im[i];
        end

        inverse           = 1;
        capture_ifft_flag = 1;
        ifft_out_cnt      = 0;
        apply_input();
        @(posedge done);
        capture_ifft_flag = 0;

        // Compare original vs reconstructed (first 10 samples)
        $display("  Sample | Original   | Reconstructed | Error");
        $display("  -------|------------|---------------|-------");
        for (i = 0; i < 10; i = i+1) begin
            $display("  %4d   | %10.6f | %13.6f | %.6f",
                i,
                0.3*$cos(2.0*3.14159265*3.0*i/NFFT)
                + 0.5*$cos(2.0*3.14159265*17.0*i/NFFT)
                + 0.2*$cos(2.0*3.14159265*50.0*i/NFFT),
                ifft_re[i],
                $abs(ifft_re[i] - (0.3*$cos(2.0*3.14159265*3.0*i/NFFT)
                + 0.5*$cos(2.0*3.14159265*17.0*i/NFFT)
                + 0.2*$cos(2.0*3.14159265*50.0*i/NFFT))));
        end
        $display("  TEST 3 COMPLETE — check errors above (expect < 0.01 for 18-bit)");

        repeat(20) @(posedge clk);
        $display("\n=== All tests complete ===");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #50_000_000;  // 50 ms timeout
        $display("TIMEOUT — simulation did not finish");
        $finish;
    end

endmodule
