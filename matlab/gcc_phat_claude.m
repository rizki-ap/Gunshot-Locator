% =========================================================
%  GCC-PHAT Implementation (No Toolbox Required)
%  Generalized Cross-Correlation with Phase Transform
%
%  Outputs:
%    tau     - Estimated Time Difference of Arrival (TDOA) in seconds
%    cc      - GCC-PHAT correlation vector
%    t_axis  - Time axis for plotting correlation
% =========================================================

clear; clc; close all;

%% --- 1. SIMULATE TWO MICROPHONE SIGNALS ---
fs      = 16000;          % Sampling frequency (Hz)
dur     = 0.5;            % Signal duration (seconds)
N       = dur * fs;       % Number of samples
t       = (0:N-1) / fs;   % Time vector

% True TDOA: microphone 2 receives signal d_samples later
true_tau    = 0.0008;             % 0.8 ms delay (e.g., ~27 cm at 340 m/s)
d_samples   = round(true_tau * fs);

% Source signal: bandlimited noise (simulates speech-like signal)
rng(42);
src = randn(1, N);

% Microphone signals with additive white noise
snr_db  = 20;
noise_power = 10^(-snr_db/10);

mic1 = src + sqrt(noise_power) * randn(1, N);
mic2 = [zeros(1, d_samples), src(1:end-d_samples)] + sqrt(noise_power) * randn(1, N);

%% --- 2. GCC-PHAT CORE FUNCTION ---
[tau, cc, t_axis] = gcc_phat(mic1, mic2, fs);

%% --- 3. DISPLAY RESULTS ---
fprintf('===== GCC-PHAT TDOA Estimation =====\n');
fprintf('True TDOA      : %.4f ms  (%d samples)\n', true_tau*1000, d_samples);
fprintf('Estimated TDOA : %.4f ms  (%.1f samples)\n', tau*1000, tau*fs);
fprintf('Error          : %.4f ms\n', abs(tau - true_tau)*1000);

%% --- 4. PLOT ---
figure('Name', 'GCC-PHAT Result', 'NumberTitle', 'off');

% Plot GCC-PHAT output
subplot(2,1,1);
plot(t_axis * 1000, cc, 'b', 'LineWidth', 1.2);
hold on;
xline(tau * 1000,       'r--', 'LineWidth', 1.5, 'Label', sprintf('Est. \\tau = %.3f ms', tau*1000));
xline(true_tau * 1000,  'g--', 'LineWidth', 1.5, 'Label', sprintf('True \\tau = %.3f ms', true_tau*1000));
xlabel('Time Delay (ms)');
ylabel('GCC-PHAT');
title('GCC-PHAT Correlation');
legend('GCC-PHAT', 'Estimated TDOA', 'True TDOA');
grid on;

% Plot microphone signals
subplot(2,1,2);
plot(t(1:200)*1000, mic1(1:200), 'b', 'DisplayName', 'Mic 1');
hold on;
plot(t(1:200)*1000, mic2(1:200), 'r--', 'DisplayName', 'Mic 2');
xlabel('Time (ms)');
ylabel('Amplitude');
title('Microphone Signals (first 200 samples)');
legend; grid on;


% =========================================================
%  GCC-PHAT Function
%  Inputs:
%    x1, x2  - input signals (row vectors, same length)
%    fs       - sampling frequency (Hz)
%    max_tau  - (optional) maximum TDOA to search (seconds)
%  Outputs:
%    tau      - estimated TDOA (seconds)
%    cc       - GCC-PHAT correlation vector
%    t_axis   - corresponding time axis (seconds)
% =========================================================
function [tau, cc, t_axis] = gcc_phat(x1, x2, fs, max_tau)

    if nargin < 4
        max_tau = [];   % No restriction by default
    end

    n  = length(x1) + length(x2) - 1;  % Linear correlation length
    nfft = 2^nextpow2(n);               % Zero-pad to next power of 2 for FFT efficiency

    % Step 1: Compute FFT of both signals
    X1 = fft(x1, nfft);
    X2 = fft(x2, nfft);

    % Step 2: Cross-power spectrum
    G = X1 .* conj(X2);

    % Step 3: PHAT weighting — normalize by magnitude (whitening)
    %         This sharpens the correlation peak
    G_phat = G ./ (abs(G) + 1e-10);    % Small epsilon avoids division by zero

    % Step 4: Inverse FFT to get correlation in time domain
    cc_full = real(ifft(G_phat, nfft));

    % Step 5: Rearrange to center zero-lag at middle (like xcorr output)
    cc = [cc_full(nfft-length(x1)+2 : end), cc_full(1 : length(x2))];

    % Step 6: Build time axis
    lags   = -(length(x1)-1) : (length(x2)-1);
    t_axis = lags / fs;

    % Step 7: Restrict search range if max_tau is specified
    if ~isempty(max_tau)
        valid = abs(t_axis) <= max_tau;
        cc_search  = cc(valid);
        t_search   = t_axis(valid);
    else
        cc_search  = cc;
        t_search   = t_axis;
    end

    % Step 8: Find peak — this is the TDOA estimate
    [~, idx] = max(cc_search);
    tau = t_search(idx);

end
