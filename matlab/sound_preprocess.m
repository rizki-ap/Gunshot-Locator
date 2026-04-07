% Read the WAV file
[y, fs] = audioread('input.wav');

% Cut 0.5 seconds from beginning and end
samples_to_cut = round(0.5 * fs);
y_cut = y(samples_to_cut+1 : end-samples_to_cut, :);

% Upsample by 4x using resample
fs_new = fs * 4;
y_upsampled = resample(y_cut, 4, 1);

% Save to new WAV file
audiowrite('output.wav', y_upsampled, fs_new);

disp('Done! File saved as output.wav');
