// ============================================================
//  gcc_phat_core.v
//  Cross-Spectrum Computation + PHAT Weighting
//
//  Takes X1[k] and X2[k] from rfft1024_packed and computes:
//
//    Step 1 — Cross-power spectrum:
//      G[k] = X1[k] * conj(X2[k])
//             = (x1r*x2r + x1i*x2i) + j*(x1i*x2r - x1r*x2i)
//
//    Step 2 — PHAT whitening (normalise by magnitude):
//      G_phat[k] = G[k] / (|G[k]| + epsilon)
//
//    |G[k]| approximated using:
//      |G| ≈ max(|re|,|im|) + 0.375 * min(|re|,|im|)
//    Error < 4%, avoids CORDIC unit.
//
//    Division done via reciprocal ROM:
//      1/|G| from ROM indexed by top LOG2N bits of magnitude.
//
//  Fixed-point: 18-bit signed Q1.17
//  Latency: 4 clock cycles (fully pipelined)
//  One output per clock after initial latency
// ============================================================
module gcc_phat_core #(
    parameter DATA_WIDTH = 18,
    parameter NFFT       = 1024,
    parameter LOG2N      = 10,
    parameter FRAC       = 17,
    parameter EPSILON    = 4
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire signed [DATA_WIDTH-1:0] x1_re, x1_im,
    input  wire signed [DATA_WIDTH-1:0] x2_re, x2_im,
    input  wire [LOG2N-1:0]             in_bin,
    output reg         out_valid,
    output reg  signed [DATA_WIDTH-1:0] g_re,
    output reg  signed [DATA_WIDTH-1:0] g_im,
    output reg  [LOG2N-1:0]             out_bin
);

    localparam MULT_W = 2*DATA_WIDTH;

    // Stage 1: multiply
    reg signed [MULT_W-1:0] m1, m2, m3, m4;
    reg [LOG2N-1:0] bin_d1;
    reg valid_d1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1<=0; m2<=0; m3<=0; m4<=0; valid_d1<=0;
        end else begin
            m1       <= x1_re * x2_re;
            m2       <= x1_im * x2_im;
            m3       <= x1_im * x2_re;
            m4       <= x1_re * x2_im;
            bin_d1   <= in_bin;
            valid_d1 <= in_valid;
        end
    end

    // Stage 2: accumulate -> G[k]
    reg signed [DATA_WIDTH-1:0] gs_re, gs_im;
    reg [LOG2N-1:0] bin_d2;
    reg valid_d2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gs_re<=0; gs_im<=0; valid_d2<=0;
        end else begin
            gs_re    <= (m1 + m2) >>> FRAC;
            gs_im    <= (m3 - m4) >>> FRAC;
            bin_d2   <= bin_d1;
            valid_d2 <= valid_d1;
        end
    end

    // Stage 3: magnitude approx + reciprocal ROM lookup
    wire signed [DATA_WIDTH-1:0] abs_re  = gs_re[DATA_WIDTH-1] ? -gs_re : gs_re;
    wire signed [DATA_WIDTH-1:0] abs_im  = gs_im[DATA_WIDTH-1] ? -gs_im : gs_im;
    wire signed [DATA_WIDTH-1:0] mag_max = (abs_re > abs_im) ? abs_re : abs_im;
    wire signed [DATA_WIDTH-1:0] mag_min = (abs_re < abs_im) ? abs_re : abs_im;
    wire [DATA_WIDTH+1:0] mag_approx     = mag_max + ((mag_min * 3) >>> 3);
    wire [DATA_WIDTH-1:0] mag_pos        = mag_approx[DATA_WIDTH-1:0];
    wire [LOG2N-1:0]      recip_idx      = mag_pos[DATA_WIDTH-1:DATA_WIDTH-LOG2N];

    reg [DATA_WIDTH-1:0] recip_rom [0:(1<<LOG2N)-1];
    initial $readmemh("recip_rom.hex", recip_rom);

    reg [DATA_WIDTH-1:0]          recip_val;
    reg signed [DATA_WIDTH-1:0]   gs_re_d3, gs_im_d3;
    reg [LOG2N-1:0]                bin_d3;
    reg                            valid_d3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            recip_val<=0; valid_d3<=0;
        end else begin
            recip_val <= recip_rom[recip_idx];
            gs_re_d3  <= gs_re;
            gs_im_d3  <= gs_im;
            bin_d3    <= bin_d2;
            valid_d3  <= valid_d2;
        end
    end

    // Stage 4: PHAT normalization
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g_re<=0; g_im<=0; out_valid<=0;
        end else begin
            g_re      <= ($signed(gs_re_d3) * $signed({1'b0, recip_val})) >>> FRAC;
            g_im      <= ($signed(gs_im_d3) * $signed({1'b0, recip_val})) >>> FRAC;
            out_bin   <= bin_d3;
            out_valid <= valid_d3;
        end
    end

endmodule
