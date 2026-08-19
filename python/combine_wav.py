import glob
import numpy as np
from scipy.io import wavfile

pattern = input("Enter WAV files [*.wav]: ") or "*.wav"

files = sorted(glob.glob(pattern))

if len(files) == 0:
    raise RuntimeError("No WAV files found")

print(f"Found {len(files)} files :")
for f in files:
    print(" ", f)

# Read first file
fs, data = wavfile.read(files[0])

data_list = [data]

# Read remaining files
for filename in files[1:]:
    fs_i, data_i = wavfile.read(filename)

    # Check sampling rate
    if fs_i != fs:
        raise ValueError(
            f"Sampling rate mismatch: {filename} has {fs_i} Hz, "
            f"expected {fs} Hz"
        )

    data_list.append(data_i)

# Check that all files have same number of samples
num_samples = len(data_list[0])

for i, data in enumerate(data_list):
    if len(data) != num_samples:
        raise ValueError(
            f"Length mismatch in {files[i]}: "
            f"{len(data)} samples, expected {num_samples}"
        )

# Combine into multichannel array
combined = np.column_stack(data_list)

print("Output shape:", combined.shape)
print("Channels:", combined.shape[1])
print("Samples:", combined.shape[0])
print("Sampling rate:", fs)

pattern = input("Enter output FileName[comb.wav]: ")

if pattern == "":
    output_file = "edes_combined.wav"
else:
    output_file = pattern

# Write multichannel WAV
wavfile.write(output_file, fs, combined)

print("Saved:", output_file)
