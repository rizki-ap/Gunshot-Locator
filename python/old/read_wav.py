import requests
from scipy.io import wavfile

x = input("How many files do you want to read [1]? ")
x = int(x) if x else 1

for i in range(x):
    filename = input(f"Enter filename {i+1}: ")

    print("Processing:", filename)

    # Your file processing code here
    samplerate, dataset = wavfile.read(filename)
    print(f"Rate: {samplerate}")
    print(f"Length: {len(dataset)}")
    print(f"Channel: {dataset.shape[1]}")
