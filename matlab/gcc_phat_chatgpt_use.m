fs = 16000; % sampling rate

% Load two microphone signals
[x, fs] = audioread('mic1.wav');
[y, fs] = audioread('mic2.wav');

% Ensure mono
x = x(:,1);
y = y(:,1);

% Run GCC-PHAT
[tau, tdoa, R] = gcc_phat(x, y, fs);

fprintf('Time delay: %.6f seconds\n', tau);

% Plot correlation
figure;
plot(R);
title('GCC-PHAT Cross-Correlation');
xlabel('Lag');
ylabel('Amplitude');

% NOTE :
% d = 0.2;           % distance between mics (meters)
% c = 343;           % speed of sound (m/s)
% theta = asin(tau * c / d);
% theta_deg = rad2deg(theta);
% fprintf('Estimated angle: %.2f degrees\n', theta_deg);