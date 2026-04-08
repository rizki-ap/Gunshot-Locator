// ============================================================
//  fft_bitrev.v
//  Bit-Reversal Permutation for 1024-point FFT
//
//  Reorders input samples x[n] into bit-reversed order
//  required by radix-2 DIT FFT.
//
//  N=1024, log2(N)=10 bits
//  Uses dual-port block RAM for efficient reordering.
//  Latency: 2*N clock cycles (write phase + read phase)
// ============================================================
module fft_bitrev #(
    parameter DATA_WIDTH = 18,
    parameter NFFT       = 1024,
    parameter LOG2N      = 10
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      start,        // pulse to begin
    input  wire signed [DATA_WIDTH-1:0] in_re,
    input  wire signed [DATA_WIDTH-1:0] in_im,

    output reg  signed [DATA_WIDTH-1:0] out_re,
    output reg  signed [DATA_WIDTH-1:0] out_im,
    output reg                       out_valid,
    output reg                       done
);

    // RAM
    reg signed [DATA_WIDTH-1:0] ram_re [0:NFFT-1];
    reg signed [DATA_WIDTH-1:0] ram_im [0:NFFT-1];

    // Bit-reversal function
    function [LOG2N-1:0] bit_reverse;
        input [LOG2N-1:0] in;
        integer i;
        begin
            for (i = 0; i < LOG2N; i = i+1)
                bit_reverse[i] = in[LOG2N-1-i];
        end
    endfunction

    // State machine
    localparam IDLE  = 2'd0,
               WRITE = 2'd1,
               READ  = 2'd2;

    reg [1:0]        state;
    reg [LOG2N-1:0]  cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            cnt       <= 0;
            out_valid <= 0;
            done      <= 0;
        end else begin
            out_valid <= 0;
            done      <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        cnt   <= 0;
                        state <= WRITE;
                    end
                end

                // Write N samples into RAM at bit-reversed addresses
                WRITE: begin
                    ram_re[bit_reverse(cnt)] <= in_re;
                    ram_im[bit_reverse(cnt)] <= in_im;
                    cnt <= cnt + 1;
                    if (cnt == NFFT-1) begin
                        cnt   <= 0;
                        state <= READ;
                    end
                end

                // Read back in natural order — output is bit-reversed input
                READ: begin
                    out_re    <= ram_re[cnt];
                    out_im    <= ram_im[cnt];
                    out_valid <= 1;
                    cnt <= cnt + 1;
                    if (cnt == NFFT-1) begin
                        done  <= 1;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
