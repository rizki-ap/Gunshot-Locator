% Specify the file name
file1 = 'ruger_1b_chan0_v0.wav'; % You can use your own file name
file2 = 'ruger_1b_chan1_v0.wav'; % You can use your own file name
file3 = 'ruger_1b_chan2_v0.wav'; % You can use your own file name
file4 = 'ruger_1b_chan3_v0.wav'; % You can use your own file name
file5 = 'ruger_1b_chan4_v0.wav'; % You can use your own file name
file6 = 'ruger_1b_chan5_v0.wav'; % You can use your own file name
file7 = 'glock3_chan6_v0.wav'; % You can use your own file name
file8 = 'glock3_mean_v0.wav'; % You can use your own file name
%D:\WORK\GunShotLocator\Dataset\edge-collected-gunshot-audio\glock_17_9mm_caliber

% Read the audio file
[y1, Fs] = audioread(file1); 
[y2, Fs] = audioread(file6); 

f_upsample = 4;
% Upsample 4x
%y1 = interp(y1,f_upsample);
%y2 = interp(y2,f_upsample);

% Play the audio using the sound function
%sound(y, Fs); 

% Optional: Plot the audio signal
% Create a time vector
info = audioinfo(file1);
t = 0:1/Fs:f_upsample*info.Duration;
t = t(round(end/3):round(2*end/3)-1); % Adjust the length to match y
y1 = y1(round(end/3):round(2*end/3));
y2 = y2(round(end/3):round(2*end/3));

[correlation_values, lags] = xcorr(y1, y2);

% Find the index corresponding to the maximum correlation value
[~, max_corr_idx] = max(abs(correlation_values)); 

% Get the lag value at that index
lag_at_max_val_samples = lags(max_corr_idx);

% Convert the lag from samples to time units
time_delay = lag_at_max_val_samples / (Fs*f_upsample);

disp(['Time delay in samples: ', num2str(lag_at_max_val_samples)]);
disp(['Time delay in seconds: ', num2str(time_delay)]);
disp(['distance in cm: ', num2str(time_delay*34300)]);

% Plot the signal
plot(t, y1, t, y2);
%plot(t, y2);
title('Audio Signal Waveform');
xlabel('Time (seconds)');
ylabel('Amplitude');

