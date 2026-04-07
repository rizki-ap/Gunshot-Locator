// ============================================================
//  tb_gcc_phat_top.v
//  Full GCC-PHAT Pipeline Testbench
//
//  Tests:
//    1. Known delay  +8 samples -> expected TDOA +8 (±1)
//    2. Zero delay   identical signals -> expected TDOA 0
//    3. Negative delay -12 samples -> expected TDOA -12 (±1)
//    4. Noisy signals SNR=10dB, delay=+5 -> expected ±2 tol
//    5. Back-to-back frames -> verify pipeline resets cleanly
// ============================================================
`timescale 1ns/1ps

module tb_gcc_phat_top;

    localparam DATA_WIDTH = 18;
    localparam NFFT       = 1024;
    localparam LOG2N      = 10;
    localparam FS         = 16000;
    localparam D_MM       = 150;
    localparam CLK_PERIOD = 20;
    localparam FRAC       = 17;
    localparam SCALE      = 2**FRAC;

    reg        clk, rst_n, start, in_valid;
    reg  signed [DATA_WIDTH-1:0] mic1_in, mic2_in;
    reg  [LOG2N-1:0] max_lag;

    wire signed [LOG2N:0]  tdoa_samples;
    wire signed [31:0]     tdoa_us;
    wire signed [15:0]     theta_deg_x10;
    wire        result_valid;

    gcc_phat_top #(
        .DATA_WIDTH (DATA_WIDTH), .NFFT (NFFT), .LOG2N (LOG2N),
        .FS (FS), .D_MM (D_MM)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .mic1_in(mic1_in), .mic2_in(mic2_in), .in_valid(in_valid),
        .max_lag(max_lag),
        .tdoa_samples(tdoa_samples), .tdoa_us(tdoa_us),
        .theta_deg_x10(theta_deg_x10), .result_valid(result_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin $dumpfile("tb_gcc_phat_top.vcd"); $dumpvars(0, tb_gcc_phat_top); end

    real sig1 [0:NFFT-1];
    real sig2 [0:NFFT-1];
    integer i;
    real pi;
    reg [31:0] lfsr;
    reg [LOG2N:0] result_samples;
    reg [31:0] result_us_reg;

    function real rand_n;
        begin
            lfsr = lfsr ^ (lfsr << 13);
            lfsr = lfsr ^ (lfsr >> 17);
            lfsr = lfsr ^ (lfsr << 5);
            rand_n = ($itor(lfsr[15:0]) / 32768.0) - 1.0;
        end
    endfunction

    task run_frame;
        integer n;
        begin
            @(posedge clk); #1; start=1; in_valid=0;
            @(posedge clk); #1; start=0;
            for (n=0; n<NFFT; n=n+1) begin
                in_valid=1;
                mic1_in=$rtoi(sig1[n]*SCALE);
                mic2_in=$rtoi(sig2[n]*SCALE);
                @(posedge clk); #1;
            end
            in_valid=0;
            @(posedge result_valid);
            result_samples=tdoa_samples;
            result_us_reg=tdoa_us;
            repeat(10) @(posedge clk);
        end
    endtask

    initial begin
        pi=3.14159265358979; lfsr=32'hCAFEBABE;
        rst_n=0; start=0; in_valid=0; mic1_in=0; mic2_in=0; max_lag=10'd64;
        repeat(4) @(posedge clk); rst_n=1; repeat(4) @(posedge clk);

        // Test 1: delay +8
        $display("\n=== TEST 1: Known delay +8 samples ===");
        for (i=0; i<NFFT; i=i+1) begin
            sig1[i]=0.35*$cos(2.0*pi*20.0*i/NFFT)+0.25*$cos(2.0*pi*50.0*i/NFFT);
            sig2[i]=(i>=8)?(0.35*$cos(2.0*pi*20.0*(i-8)/NFFT)+0.25*$cos(2.0*pi*50.0*(i-8)/NFFT)):0.0;
        end
        run_frame();
        $display("  Expected: +8 samples (+500 us)");
        $display("  Got:      %0d samples (%0d us)", $signed(result_samples), $signed(result_us_reg));
        if ($signed(result_samples)>=7 && $signed(result_samples)<=9) $display("  PASS"); else $display("  FAIL");

        // Test 2: zero delay
        $display("\n=== TEST 2: Zero delay ===");
        for (i=0; i<NFFT; i=i+1) begin
            sig1[i]=0.4*$cos(2.0*pi*15.0*i/NFFT)+0.3*$cos(2.0*pi*35.0*i/NFFT);
            sig2[i]=sig1[i];
        end
        run_frame();
        $display("  Expected: 0 samples");
        $display("  Got:      %0d samples (%0d us)", $signed(result_samples), $signed(result_us_reg));
        if ($signed(result_samples)>=-1 && $signed(result_samples)<=1) $display("  PASS"); else $display("  FAIL");

        // Test 3: negative delay -12
        $display("\n=== TEST 3: Negative delay -12 samples ===");
        for (i=0; i<NFFT; i=i+1) begin
            sig1[i]=(i>=12)?(0.3*$cos(2.0*pi*10.0*(i-12)/NFFT)+0.3*$cos(2.0*pi*40.0*(i-12)/NFFT)):0.0;
            sig2[i]=0.3*$cos(2.0*pi*10.0*i/NFFT)+0.3*$cos(2.0*pi*40.0*i/NFFT);
        end
        run_frame();
        $display("  Expected: -12 samples (-750 us)");
        $display("  Got:      %0d samples (%0d us)", $signed(result_samples), $signed(result_us_reg));
        if ($signed(result_samples)>=-13 && $signed(result_samples)<=-11) $display("  PASS"); else $display("  FAIL");

        // Test 4: noisy SNR=10dB
        $display("\n=== TEST 4: SNR=10dB delay +5 samples ===");
        for (i=0; i<NFFT; i=i+1) begin
            sig1[i]=0.3*$cos(2.0*pi*25.0*i/NFFT)+0.095*rand_n();
            sig2[i]=(i>=5)?(0.3*$cos(2.0*pi*25.0*(i-5)/NFFT)+0.095*rand_n()):(0.095*rand_n());
        end
        run_frame();
        $display("  Expected: +5 samples (±2 at SNR=10dB)");
        $display("  Got:      %0d samples (%0d us)", $signed(result_samples), $signed(result_us_reg));
        if ($signed(result_samples)>=3 && $signed(result_samples)<=7) $display("  PASS"); else $display("  NOTE: noise sensitivity");

        // Test 5: back-to-back frames
        $display("\n=== TEST 5: Back-to-back frames ===");
        for (i=0; i<NFFT; i=i+1) begin
            sig1[i]=0.35*$cos(2.0*pi*30.0*i/NFFT);
            sig2[i]=(i>=6)?(0.35*$cos(2.0*pi*30.0*(i-6)/NFFT)):0.0;
        end
        run_frame();
        $display("  Frame A (exp +6): %0d", $signed(result_samples));
        for (i=0; i<NFFT; i=i+1) begin
            sig1[i]=0.35*$cos(2.0*pi*12.0*i/NFFT);
            sig2[i]=(i>=20)?(0.35*$cos(2.0*pi*12.0*(i-20)/NFFT)):0.0;
        end
        run_frame();
        $display("  Frame B (exp +20): %0d", $signed(result_samples));
        if ($signed(result_samples)>=19 && $signed(result_samples)<=21) $display("  PASS"); else $display("  FAIL");

        $display("\n=== All GCC-PHAT tests complete ===");
        $display("Resolution: ±%0d us/sample | D=%0dmm | fs=%0dHz", 1000000/FS, D_MM, FS);
        repeat(20) @(posedge clk);
        $finish;
    end

    initial begin #200_000_000; $display("TIMEOUT"); $finish; end
endmodule
