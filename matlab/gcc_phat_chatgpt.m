function [tau, tdoa, R] = gcc_phat_chatgpt(x, y, fs)
% GCC-PHAT implementation
% x, y  : input signals (same length)
% fs    : sampling frequency
% tau   : time delay (seconds)
% tdoa  : lag index
% R     : cross-correlation (PHAT)

    N = length(x) + length(y);

    % FFT of both signals
    X = fft(x, N);
    Y = fft(y, N);

    % Cross-power spectrum
    G = X .* conj(Y);

    % PHAT weighting (normalize magnitude)
    G = G ./ (abs(G) + eps);

    % Inverse FFT to get cross-correlation
    R = real(ifft(G));

    % Shift zero lag to center
    R = fftshift(R);

    % Find peak (TDOA)
    [~, idx] = max(R);

    % Compute lag
    max_shift = floor(N/2);
    tdoa = idx - max_shift - 1;

    % Convert to time delay
    tau = tdoa / fs;
end