// ============================================================
//  peak_detector.v
//  Argmax over the GCC-PHAT correlation vector cc[n]
//
//  Receives cc[n] samples one per clock from the IFFT output.
//  Tracks the running maximum and its index.
//  After all N samples, outputs the lag index of the peak.
//
//  Also enforces a valid lag window: only searches within
//  ±max_lag samples of centre (N/2). This removes spurious
//  peaks outside the physically possible TDOA range.
//
//  Ports:
//    in_valid   - qualifies cc input
//    cc_re      - real part of IFFT output (imaginary discarded)
//    in_idx     - sample index from IFFT (0..N-1)
//    max_lag    - maximum lag to search (set from HPS register)
//    done_in    - pulse from IFFT when last sample arrives
//    peak_idx   - lag index of correlation peak (0..N-1)
//    peak_valid - single-cycle pulse: peak_idx is valid
// ============================================================
module peak_detector #(
    parameter DATA_WIDTH = 18,
    parameter NFFT       = 1024,
    parameter LOG2N      = 10
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire signed [DATA_WIDTH-1:0] cc_re,
    input  wire [LOG2N-1:0]             in_idx,
    input  wire [LOG2N-1:0]             max_lag,    // e.g. 64 for ±4ms at 16kHz
    input  wire        done_in,
    output reg  [LOG2N-1:0]             peak_idx,
    output reg         peak_valid
);

    // Centre of correlation output is at index N/2
    localparam CENTRE = NFFT / 2;

    reg signed [DATA_WIDTH-1:0] max_val;
    reg [LOG2N-1:0]              max_idx;
    reg                          searching;

    // Lag of current sample: signed distance from centre
    wire signed [LOG2N:0] lag = $signed({1'b0, in_idx}) - $signed({1'b0, CENTRE[LOG2N-1:0]});
    wire signed [LOG2N:0] lag_abs = lag[LOG2N] ? -lag : lag;

    // Is this sample within the valid search window?
    wire in_window = (lag_abs <= $signed({1'b0, max_lag}));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_val    <= {1'b1, {(DATA_WIDTH-1){1'b0}}};  // most negative
            max_idx    <= 0;
            searching  <= 0;
            peak_valid <= 0;
            peak_idx   <= 0;
        end else begin
            peak_valid <= 0;

            if (in_valid && in_window) begin
                searching <= 1;
                if (cc_re > max_val) begin
                    max_val <= cc_re;
                    max_idx <= in_idx;
                end
            end

            if (done_in && searching) begin
                peak_idx   <= max_idx;
                peak_valid <= 1;
                // Reset for next frame
                max_val   <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
                searching <= 0;
            end
        end
    end

endmodule


// ============================================================
//  tdoa_calc.v
//  Convert peak lag index to TDOA in fixed-point seconds
//
//  tau = (peak_idx - N/2) / fs
//
//  Since fs is constant (e.g. 16000 Hz), we precompute:
//    tau_scale = 1/fs in Q0.32 fixed-point
//    tau_us    = (lag_samples * 1e6) / fs  (integer microseconds)
//
//  Outputs:
//    tdoa_samples  - signed lag in samples (range ±max_lag)
//    tdoa_us       - signed TDOA in microseconds (for HPS/display)
//    tdoa_valid    - single-cycle pulse
//
//  Also outputs estimated sound source angle assuming a
//  linear microphone pair:
//    theta = arcsin( tau * c / d )
//  Approximated for small angles as:
//    theta_deg ≈ (tau_us * 340 * 180) / (d_mm * pi * 1000)
//  where c=340 m/s, d_mm = mic spacing in mm (set by parameter)
// ============================================================
module tdoa_calc #(
    parameter DATA_WIDTH  = 18,
    parameter NFFT        = 1024,
    parameter LOG2N       = 10,
    parameter FS          = 16000,  // sampling frequency Hz
    parameter D_MM        = 150     // mic spacing in mm (15 cm)
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        peak_valid,
    input  wire [LOG2N-1:0] peak_idx,

    output reg  signed [LOG2N:0]   tdoa_samples,  // signed lag, range ±512
    output reg  signed [31:0]      tdoa_us,        // signed microseconds
    output reg  signed [15:0]      theta_deg_x10,  // angle * 10 (1 decimal place)
    output reg                     tdoa_valid
);

    localparam CENTRE    = NFFT / 2;                   // 512
    localparam US_SCALE  = 1000000;                    // 1e6 for microseconds
    // Angle scale: (c * 180 * 1e6) / (d_mm * pi * 1000 * fs)
    // Numerator:   340 * 180 * 1e6 = 61,200,000,000
    // Denominator: 150 * 3.14159 * 1000 * 16000 = 7,539,816,000
    // => 1 sample lag ≈ 8.12 degrees * 10 = 81.2 (round to 81)
    localparam ANGLE_SCALE = (340 * 180 * 10) / (D_MM * 314159 / 100000 / FS);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tdoa_samples  <= 0;
            tdoa_us       <= 0;
            theta_deg_x10 <= 0;
            tdoa_valid    <= 0;
        end else begin
            tdoa_valid <= 0;

            if (peak_valid) begin
                // Signed lag in samples
                tdoa_samples  <= $signed({1'b0, peak_idx}) - CENTRE;

                // TDOA in microseconds: lag * 1e6 / fs
                tdoa_us       <= ($signed({1'b0, peak_idx}) - CENTRE) * (US_SCALE / FS);

                // Angle approximation (small angle, linear array)
                theta_deg_x10 <= ($signed({1'b0, peak_idx}) - CENTRE) * ANGLE_SCALE;

                tdoa_valid    <= 1;
            end
        end
    end

endmodule
