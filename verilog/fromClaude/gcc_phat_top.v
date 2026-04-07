// ============================================================
//  gcc_phat_top.v
//  Complete GCC-PHAT Pipeline Top-Level
//
//  Integrates all stages:
//    1. rfft1024_packed  — one real FFT for two mic channels
//    2. gcc_phat_core    — cross-spectrum + PHAT weighting
//    3. fft1024 (IFFT)   — convert back to time domain
//    4. peak_detector    — argmax over correlation
//    5. tdoa_calc        — lag → time/angle conversion
//
//  Resource saving vs naive two-channel complex FFT:
//    - 50% fewer FFT engines (1 instead of 2)
//    - Only 513 bins processed (not 1024)
//    - Single IFFT on 513-bin PHAT output
//
//  Ports:
//    clk           - system clock (50 MHz on Cyclone V)
//    rst_n         - active-low reset
//    start         - pulse to begin new frame
//    mic1_in       - 18-bit PCM sample, microphone 1
//    mic2_in       - 18-bit PCM sample, microphone 2
//    in_valid      - qualifies mic samples (one per clock)
//    max_lag       - max TDOA lag to search (from HPS register)
//    tdoa_samples  - TDOA result in samples (signed)
//    tdoa_us       - TDOA result in microseconds (signed)
//    theta_deg_x10 - estimated DOA angle * 10 degrees
//    result_valid  - single-cycle pulse: outputs are valid
// ============================================================
module gcc_phat_top #(
    parameter DATA_WIDTH = 18,
    parameter NFFT       = 1024,
    parameter LOG2N      = 10,
    parameter FS         = 16000,
    parameter D_MM       = 150
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [DATA_WIDTH-1:0] mic1_in,
    input  wire signed [DATA_WIDTH-1:0] mic2_in,
    input  wire        in_valid,
    input  wire [LOG2N-1:0] max_lag,

    output wire signed [LOG2N:0]  tdoa_samples,
    output wire signed [31:0]     tdoa_us,
    output wire signed [15:0]     theta_deg_x10,
    output wire        result_valid
);

    // -------------------------------------------------------
    // Stage 1: Real-packed FFT
    // -------------------------------------------------------
    wire signed [DATA_WIDTH-1:0] x1_re, x1_im, x2_re, x2_im;
    wire [LOG2N-1:0]              fft_bin;
    wire        fft_out_valid, fft_done;

    rfft1024_packed #(
        .DATA_WIDTH    (DATA_WIDTH),
        .TWIDDLE_WIDTH (DATA_WIDTH),
        .NFFT          (NFFT),
        .LOG2N         (LOG2N)
    ) u_rfft (
        .clk       (clk), .rst_n    (rst_n), .start    (start),
        .mic1_in   (mic1_in), .mic2_in  (mic2_in), .in_valid (in_valid),
        .x1_re     (x1_re),   .x1_im    (x1_im),
        .x2_re     (x2_re),   .x2_im    (x2_im),
        .out_bin   (fft_bin), .out_valid(fft_out_valid), .done(fft_done)
    );

    // -------------------------------------------------------
    // Stage 2: GCC-PHAT core (cross-spectrum + PHAT weight)
    // -------------------------------------------------------
    wire signed [DATA_WIDTH-1:0] g_re, g_im;
    wire [LOG2N-1:0]              phat_bin;
    wire        phat_valid;

    gcc_phat_core #(
        .DATA_WIDTH (DATA_WIDTH),
        .NFFT       (NFFT),
        .LOG2N      (LOG2N)
    ) u_phat (
        .clk       (clk), .rst_n    (rst_n),
        .in_valid  (fft_out_valid),
        .x1_re     (x1_re),   .x1_im    (x1_im),
        .x2_re     (x2_re),   .x2_im    (x2_im),
        .in_bin    (fft_bin),
        .out_valid (phat_valid),
        .g_re      (g_re),    .g_im     (g_im),
        .out_bin   (phat_bin)
    );

    // -------------------------------------------------------
    // Stage 3: IFFT of PHAT-weighted spectrum
    //   G_phat[k] is only defined for k=0..N/2 (513 bins).
    //   For real-valued correlation output we must feed the
    //   full N-bin conjugate-symmetric spectrum into the IFFT.
    //   We reconstruct the missing bins on-the-fly:
    //     G[N-k] = conj(G[k])  for k=1..N/2-1
    //   This is done in the bin_expand module below.
    // -------------------------------------------------------

    // Bin expansion: 513 bins -> 1024 bins (reconstruct symmetry)
    wire signed [DATA_WIDTH-1:0] ifft_in_re, ifft_in_im;
    wire        ifft_in_valid, ifft_start;
    wire [LOG2N-1:0] expand_bin;

    bin_expand #(
        .DATA_WIDTH (DATA_WIDTH),
        .NFFT       (NFFT),
        .LOG2N      (LOG2N)
    ) u_expand (
        .clk       (clk), .rst_n    (rst_n),
        .in_valid  (phat_valid),
        .in_re     (g_re),    .in_im    (g_im),
        .in_bin    (phat_bin),
        .out_valid (ifft_in_valid),
        .out_re    (ifft_in_re), .out_im (ifft_in_im),
        .out_bin   (expand_bin),
        .ifft_start(ifft_start)
    );

    wire signed [DATA_WIDTH-1:0] cc_re, cc_im;
    wire        cc_valid, cc_done;

    fft1024 #(
        .DATA_WIDTH    (DATA_WIDTH),
        .TWIDDLE_WIDTH (DATA_WIDTH),
        .NFFT          (NFFT),
        .LOG2N         (LOG2N)
    ) u_ifft (
        .clk       (clk), .rst_n   (rst_n),
        .inverse   (1'b1),
        .start     (ifft_start),
        .in_re     (ifft_in_re), .in_im (ifft_in_im),
        .in_valid  (ifft_in_valid),
        .out_re    (cc_re),  .out_im (cc_im),
        .out_valid (cc_valid), .done (cc_done)
    );

    // -------------------------------------------------------
    // Stage 4: Peak detector
    // -------------------------------------------------------
    wire [LOG2N-1:0] peak_idx;
    wire        peak_valid_pulse;
    reg  [LOG2N-1:0] cc_idx;

    // Track output sample index from IFFT
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)       cc_idx <= 0;
        else if (cc_valid) cc_idx <= cc_idx + 1;
        else if (cc_done)  cc_idx <= 0;
    end

    peak_detector #(
        .DATA_WIDTH (DATA_WIDTH),
        .NFFT       (NFFT),
        .LOG2N      (LOG2N)
    ) u_peak (
        .clk        (clk), .rst_n    (rst_n),
        .in_valid   (cc_valid),
        .cc_re      (cc_re),
        .in_idx     (cc_idx),
        .max_lag    (max_lag),
        .done_in    (cc_done),
        .peak_idx   (peak_idx),
        .peak_valid (peak_valid_pulse)
    );

    // -------------------------------------------------------
    // Stage 5: TDOA calculation
    // -------------------------------------------------------
    tdoa_calc #(
        .DATA_WIDTH (DATA_WIDTH),
        .NFFT       (NFFT),
        .LOG2N      (LOG2N),
        .FS         (FS),
        .D_MM       (D_MM)
    ) u_tdoa (
        .clk          (clk), .rst_n      (rst_n),
        .peak_valid   (peak_valid_pulse),
        .peak_idx     (peak_idx),
        .tdoa_samples (tdoa_samples),
        .tdoa_us      (tdoa_us),
        .theta_deg_x10(theta_deg_x10),
        .tdoa_valid   (result_valid)
    );

endmodule


// ============================================================
//  bin_expand.v
//  Reconstruct full N-bin conjugate-symmetric spectrum
//  from the 513 unique bins output by rfft1024_packed.
//
//  Algorithm:
//    1. Store 513 bins in RAM as they arrive
//    2. When last bin (N/2=512) received, stream out:
//       k=0..512        : G[k]           (as stored)
//       k=513..1023     : conj(G[N-k])   (reconstructed)
//    3. Pulse ifft_start before streaming begins
// ============================================================
module bin_expand #(
    parameter DATA_WIDTH = 18,
    parameter NFFT       = 1024,
    parameter LOG2N      = 10,
    parameter HALF       = NFFT/2
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire signed [DATA_WIDTH-1:0] in_re, in_im,
    input  wire [LOG2N-1:0]             in_bin,

    output reg         out_valid,
    output reg  signed [DATA_WIDTH-1:0] out_re, out_im,
    output reg  [LOG2N-1:0]             out_bin,
    output reg         ifft_start
);

    // RAM to store 513 bins
    reg signed [DATA_WIDTH-1:0] ram_re [0:HALF];
    reg signed [DATA_WIDTH-1:0] ram_im [0:HALF];

    localparam S_STORE  = 2'd0,
               S_STREAM = 2'd1,
               S_DONE   = 2'd2;

    reg [1:0]       state;
    reg [LOG2N-1:0] out_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_STORE;
            out_cnt    <= 0;
            out_valid  <= 0;
            ifft_start <= 0;
        end else begin
            out_valid  <= 0;
            ifft_start <= 0;

            case (state)
                // ------------------------------------------
                // Store incoming bins into RAM
                S_STORE: begin
                    if (in_valid) begin
                        ram_re[in_bin] <= in_re;
                        ram_im[in_bin] <= in_im;
                        // When last bin (N/2) arrives, begin streaming
                        if (in_bin == HALF) begin
                            out_cnt    <= 0;
                            ifft_start <= 1;
                            state      <= S_STREAM;
                        end
                    end
                end

                // ------------------------------------------
                // Stream full N bins to IFFT
                S_STREAM: begin
                    out_valid <= 1;
                    out_bin   <= out_cnt;

                    if (out_cnt <= HALF) begin
                        // First half: bins 0..512 as-is
                        out_re <= ram_re[out_cnt];
                        out_im <= ram_im[out_cnt];
                    end else begin
                        // Second half: conjugate of G[N-k]
                        // N-k for k=513..1023 maps to 511..1
                        out_re <=  ram_re[NFFT - out_cnt];
                        out_im <= -ram_im[NFFT - out_cnt];  // conjugate
                    end

                    out_cnt <= out_cnt + 1;

                    if (out_cnt == NFFT - 1) begin
                        state <= S_STORE;  // ready for next frame
                    end
                end

                default: state <= S_STORE;
            endcase
        end
    end

endmodule
