in0 = 'ruger_1b_chan0_v0.wav'; % You can use your own file name
in1 = 'ruger_1b_chan1_v0.wav'; % You can use your own file name
in2 = 'ruger_1b_chan2_v0.wav'; % You can use your own file name
in3 = 'ruger_1b_chan3_v0.wav'; % You can use your own file name
in4 = 'ruger_1b_chan4_v0.wav'; % You can use your own file name
in10 = 'glock3_chan0_v0.wav';
in11 = 'glock3_chan1_v0.wav';
in12 = 'glock3_chan2_v0.wav';
in13 = 'glock3_chan3_v0.wav';

%outf = 'ruger_1b_ch4.wav'; 
outf = 'glock3_ch3.wav'; 

% Read the WAV file
[y, fs] = audioread(in13);

%t = 0:length(y)-1/fs;
%figure(1); plot(t, y);

% Cut start=1.5s and end=0.25s
samples_start_cut = round(1.5* fs);
samples_end_cut = round(0.25* fs);
y_cut = y(samples_start_cut+1 : end-samples_end_cut, :);

t = 0:length(y_cut)-1/fs;
figure(1);
plot(t, y_cut);

% Upsample by 4x using resample
fs_new = fs * 4;
y_upsampled = resample(y_cut, 4, 1);

t = 0:length(y_upsampled)-1/fs;
figure(2); plot(t, y_upsampled);

% Save to new WAV file
audiowrite(outf, y_upsampled, fs_new);

disp('Done! File saved as output.wav');
