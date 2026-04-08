// ============================================================
//  fft1024.v
//  1024-point Radix-2 DIT FFT / IFFT Top-Level
//
//  Architecture: Iterative, single butterfly, 10 stages
//  Each stage processes N/2 = 512 butterfly operations
//  Total compute cycles per transform: 10 * 512 = 5120
//  Plus bit-reversal overhead: 2 * 1024 = 2048 cycles
//  Total latency: ~7168 clock cycles @ fs_clk
//
//  Fixed-point: 18-bit signed Q1.17
//  Twiddle ROM: 512 entries (cos/sin), loaded from hex file
//
//  IFFT mode: conjugate twiddle factors + output scale by N
//
//  Ports:
//    clk       - system clock
//    rst_n     - active-low reset
//    inverse   - 0=FFT, 1=IFFT
//    start     - pulse high for 1 cycle to begin
//    in_re/im  - time-domain input (presented over N cycles)
//    in_valid  - qualify input samples
//    out_re/im - frequency-domain output (presented over N cycles)
//    out_valid - qualifies output samples
//    done      - single-cycle pulse when transform is complete
// ============================================================
module fft1024 #(
    parameter DATA_WIDTH    = 18,
    parameter TWIDDLE_WIDTH = 18,
    parameter NFFT          = 1024,
    parameter LOG2N         = 10
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        inverse,      // 0 = FFT, 1 = IFFT
    input  wire        start,
    input  wire signed [DATA_WIDTH-1:0] in_re,
    input  wire signed [DATA_WIDTH-1:0] in_im,
    input  wire        in_valid,

    output wire signed [DATA_WIDTH-1:0] out_re,
    output wire signed [DATA_WIDTH-1:0] out_im,
    output wire        out_valid,
    output wire        done
);

    // -------------------------------------------------------
    // Internal RAM: two ping-pong buffers A and B
    // Stage k reads from A, writes to B (or vice versa)
    // -------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] buf_re [0:1][0:NFFT-1];
    reg signed [DATA_WIDTH-1:0] buf_im [0:1][0:NFFT-1];
    reg active_buf; // which buffer is currently being written to

    // -------------------------------------------------------
    // Input shift register: collect N samples
    // -------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] in_buf_re [0:NFFT-1];
    reg signed [DATA_WIDTH-1:0] in_buf_im [0:NFFT-1];
    reg [LOG2N-1:0] in_cnt;
    reg             input_done;

    // Bit-reversal wiring
    wire signed [DATA_WIDTH-1:0] bitrev_out_re, bitrev_out_im;
    wire        bitrev_out_valid, bitrev_done;
    reg         bitrev_start;

    // Temporary wires for input into bit-reversal
    reg signed [DATA_WIDTH-1:0] br_in_re, br_in_im;

    // -------------------------------------------------------
    // State Machine
    // -------------------------------------------------------
    localparam  S_IDLE     = 3'd0,
                S_INPUT    = 3'd1,
                S_BITREV   = 3'd2,
                S_COMPUTE  = 3'd3,
                S_OUTPUT   = 3'd4;

    reg [2:0]        state;
    reg [LOG2N-1:0]  stage;       // current FFT stage (0..9)
    reg [LOG2N-1:0]  bf_idx;      // butterfly index within stage (0..511)
    reg [LOG2N-1:0]  out_cnt;     // output sample counter
    reg              done_reg;
    reg              out_valid_reg;
    reg signed [DATA_WIDTH-1:0] out_re_reg, out_im_reg;

    assign done      = done_reg;
    assign out_valid = out_valid_reg;
    assign out_re    = out_re_reg;
    assign out_im    = out_im_reg;

    // -------------------------------------------------------
    // Twiddle ROM (512 entries, one read port)
    // -------------------------------------------------------
    reg  [$clog2(NFFT/2)-1:0]      tw_addr;
    wire signed [TWIDDLE_WIDTH-1:0] tw_cos, tw_sin;

    fft_twiddle_rom #(
        .DATA_WIDTH (TWIDDLE_WIDTH),
        .NFFT       (NFFT)
    ) u_rom (
        .clk     (clk),
        .addr    (tw_addr),
        .cos_out (tw_cos),
        .sin_out (tw_sin)
    );

    // For IFFT: conjugate twiddle (negate sine)
    wire signed [TWIDDLE_WIDTH-1:0] w_re_use, w_im_use;
    assign w_re_use = tw_cos;
    assign w_im_use = inverse ? -tw_sin : tw_sin;

    // -------------------------------------------------------
    // Butterfly unit
    // -------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] bf_a_re, bf_a_im;
    reg signed [DATA_WIDTH-1:0] bf_b_re, bf_b_im;
    wire signed [DATA_WIDTH-1:0] bf_p_re, bf_p_im;
    wire signed [DATA_WIDTH-1:0] bf_q_re, bf_q_im;

    fft_butterfly #(
        .DATA_WIDTH    (DATA_WIDTH),
        .TWIDDLE_WIDTH (TWIDDLE_WIDTH)
    ) u_bf (
        .clk   (clk),
        .rst_n (rst_n),
        .a_re  (bf_a_re),   .a_im  (bf_a_im),
        .b_re  (bf_b_re),   .b_im  (bf_b_im),
        .w_re  (w_re_use),  .w_im  (w_im_use),
        .p_re  (bf_p_re),   .p_im  (bf_p_im),
        .q_re  (bf_q_re),   .q_im  (bf_q_im)
    );

    // Bit-reversal module
    fft_bitrev #(
        .DATA_WIDTH (DATA_WIDTH),
        .NFFT       (NFFT),
        .LOG2N      (LOG2N)
    ) u_bitrev (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (bitrev_start),
        .in_re     (br_in_re),
        .in_im     (br_in_im),
        .out_re    (bitrev_out_re),
        .out_im    (bitrev_out_im),
        .out_valid (bitrev_out_valid),
        .done      (bitrev_done)
    );

    // -------------------------------------------------------
    // Butterfly index calculation
    // For stage s, the butterfly span = 2^s
    // Upper index: i, Lower index: i + span/2
    // Twiddle index: (i mod (span/2)) * (N/span)
    // -------------------------------------------------------
    reg [LOG2N-1:0] span_half;   // 2^(stage-1)
    reg [LOG2N-1:0] upper_idx;
    reg [LOG2N-1:0] lower_idx;
    reg [LOG2N-1:0] tw_k;

    // Pipeline delay tracking for butterfly (3 cycle latency)
    reg [LOG2N-1:0] upper_d1, lower_d1, upper_d2, lower_d2, upper_d3, lower_d3;
    reg             bf_valid_d1, bf_valid_d2, bf_valid_d3;

    // -------------------------------------------------------
    // Main FSM
    // -------------------------------------------------------
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            in_cnt        <= 0;
            input_done    <= 0;
            stage         <= 0;
            bf_idx        <= 0;
            out_cnt       <= 0;
            active_buf    <= 0;
            done_reg      <= 0;
            out_valid_reg <= 0;
            bitrev_start  <= 0;
            bf_valid_d1   <= 0;
            bf_valid_d2   <= 0;
            bf_valid_d3   <= 0;
        end else begin
            done_reg      <= 0;
            out_valid_reg <= 0;
            bitrev_start  <= 0;

            case (state)

                // ---------------------------------
                S_IDLE: begin
                    if (start) begin
                        in_cnt <= 0;
                        state  <= S_INPUT;
                    end
                end

                // ---------------------------------
                // Collect N input samples
                S_INPUT: begin
                    if (in_valid) begin
                        in_buf_re[in_cnt] <= in_re;
                        in_buf_im[in_cnt] <= in_im;
                        in_cnt <= in_cnt + 1;
                        if (in_cnt == NFFT-1) begin
                            in_cnt       <= 0;
                            bitrev_start <= 1;
                            state        <= S_BITREV;
                        end
                    end
                end

                // ---------------------------------
                // Bit-reversal reorder into buf[0]
                S_BITREV: begin
                    br_in_re <= in_buf_re[in_cnt];
                    br_in_im <= in_buf_im[in_cnt];
                    in_cnt   <= in_cnt + 1;

                    if (bitrev_out_valid) begin
                        buf_re[0][in_cnt] <= bitrev_out_re;
                        buf_im[0][in_cnt] <= bitrev_out_im;
                    end

                    if (bitrev_done) begin
                        stage      <= 0;
                        bf_idx     <= 0;
                        active_buf <= 0;
                        state      <= S_COMPUTE;
                    end
                end

                // ---------------------------------
                // FFT computation: 10 stages x 512 butterflies
                S_COMPUTE: begin
                    // Compute span and indices for current stage
                    span_half <= (1 << stage);           // 2^stage = half-span
                    upper_idx <= (bf_idx / span_half) * (span_half << 1) + (bf_idx % span_half);
                    lower_idx <= upper_idx + span_half;
                    tw_k      <= (bf_idx % span_half) * (NFFT >> (stage+1));
                    tw_addr   <= tw_k;

                    // Feed butterfly
                    bf_a_re <= buf_re[active_buf][upper_idx];
                    bf_a_im <= buf_im[active_buf][upper_idx];
                    bf_b_re <= buf_re[active_buf][lower_idx];
                    bf_b_im <= buf_im[active_buf][lower_idx];

                    // Pipeline: track write-back addresses
                    upper_d1    <= upper_idx;  lower_d1    <= lower_idx;
                    upper_d2    <= upper_d1;   lower_d2    <= lower_d1;
                    upper_d3    <= upper_d2;   lower_d3    <= lower_d2;
                    bf_valid_d1 <= 1;
                    bf_valid_d2 <= bf_valid_d1;
                    bf_valid_d3 <= bf_valid_d2;

                    // Write back butterfly result after 3-cycle latency
                    if (bf_valid_d3) begin
                        buf_re[~active_buf][upper_d3] <= bf_p_re;
                        buf_im[~active_buf][upper_d3] <= bf_p_im;
                        buf_re[~active_buf][lower_d3] <= bf_q_re;
                        buf_im[~active_buf][lower_d3] <= bf_q_im;
                    end

                    bf_idx <= bf_idx + 1;

                    if (bf_idx == NFFT/2 - 1) begin
                        bf_idx     <= 0;
                        active_buf <= ~active_buf;
                        stage      <= stage + 1;

                        if (stage == LOG2N - 1) begin
                            out_cnt <= 0;
                            state   <= S_OUTPUT;
                        end
                    end
                end

                // ---------------------------------
                // Stream output samples
                S_OUTPUT: begin
                    out_re_reg    <= buf_re[active_buf][out_cnt];
                    out_im_reg    <= buf_im[active_buf][out_cnt];
                    out_valid_reg <= 1;
                    out_cnt       <= out_cnt + 1;

                    if (out_cnt == NFFT-1) begin
                        done_reg <= 1;
                        state    <= S_IDLE;
                    end
                end

            endcase
        end
    end

endmodule
