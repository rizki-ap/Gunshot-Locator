// ============================================================
//  fft_butterfly.v
//  Single Radix-2 DIT Butterfly Unit
//
//  Computes:
//    P = A + W*B
//    Q = A - W*B
//
//  Fixed-point arithmetic, 18-bit signed inputs/outputs
//  Twiddle multiplication uses 18x18 -> truncated to 18-bit
//
//  Latency: 3 clock cycles (pipeline registered)
// ============================================================
module fft_butterfly #(
    parameter DATA_WIDTH   = 18,   // Bit width of real/imag parts
    parameter TWIDDLE_WIDTH = 18   // Twiddle factor width (Q1.17)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Input A (upper butterfly input)
    input  wire signed [DATA_WIDTH-1:0]  a_re,
    input  wire signed [DATA_WIDTH-1:0]  a_im,

    // Input B (lower butterfly input)
    input  wire signed [DATA_WIDTH-1:0]  b_re,
    input  wire signed [DATA_WIDTH-1:0]  b_im,

    // Twiddle factor W = cos - j*sin
    input  wire signed [TWIDDLE_WIDTH-1:0] w_re,  // cos(2*pi*k/N)
    input  wire signed [TWIDDLE_WIDTH-1:0] w_im,  // -sin(2*pi*k/N)

    // Outputs P and Q
    output reg  signed [DATA_WIDTH-1:0]  p_re,
    output reg  signed [DATA_WIDTH-1:0]  p_im,
    output reg  signed [DATA_WIDTH-1:0]  q_re,
    output reg  signed [DATA_WIDTH-1:0]  q_im
);

    // -------------------------------------------------------
    // Stage 1: Complex multiply W*B
    //   (w_re + j*w_im)(b_re + j*b_im)
    //   = w_re*b_re - w_im*b_im  + j*(w_re*b_im + w_im*b_re)
    // -------------------------------------------------------
    localparam MULT_WIDTH = DATA_WIDTH + TWIDDLE_WIDTH; // 36-bit products

    reg signed [MULT_WIDTH-1:0] m1, m2, m3, m4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1 <= 0; m2 <= 0; m3 <= 0; m4 <= 0;
        end else begin
            m1 <= w_re * b_re;
            m2 <= w_im * b_im;
            m3 <= w_re * b_im;
            m4 <= w_im * b_re;
        end
    end

    // -------------------------------------------------------
    // Stage 2: Accumulate products, truncate to DATA_WIDTH
    //   Q1.17 * Q1.17 = Q2.34 -> shift right 17 to get Q1.17
    // -------------------------------------------------------
    localparam FRAC = TWIDDLE_WIDTH - 1; // 17

    reg signed [DATA_WIDTH-1:0] wb_re, wb_im;
    reg signed [DATA_WIDTH-1:0] a_re_d, a_im_d; // delayed A to match latency

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_re   <= 0;
            wb_im   <= 0;
            a_re_d  <= 0;
            a_im_d  <= 0;
        end else begin
            wb_re  <= (m1 - m2) >>> FRAC;
            wb_im  <= (m3 + m4) >>> FRAC;
            a_re_d <= a_re;  // 1-cycle delay to match mult pipeline
            a_im_d <= a_im;
        end
    end

    // -------------------------------------------------------
    // Stage 3: Butterfly add/subtract
    // -------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_re <= 0; p_im <= 0;
            q_re <= 0; q_im <= 0;
        end else begin
            p_re <= (a_re_d + wb_re) >>> 1;  // scale by 0.5 to prevent overflow
            p_im <= (a_im_d + wb_im) >>> 1;
            q_re <= (a_re_d - wb_re) >>> 1;
            q_im <= (a_im_d - wb_im) >>> 1;
        end
    end

endmodule
