""" 
Process 
1. find onsets in audio file
2. pick random points in initial onsets to randomly stretch or squeeze the original audio signal 
3. find onsets again (required? yes because we do not stretch/squeeze for all onsets, only some) 
"""

import librosa
import numpy as np
import soundfile as sf
import random
# from pydub import AudioSegment

# Load the MP3 file
input_mp3 = "./audio/problem-with-society.mp3"
y, sr = librosa.load(input_mp3, sr=None)

# Detect Initial Onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
print('Detected {} onsets'.format(len(onset_frames)))

# Randomly Stretch or Squeeze Audio at Some Onsets
num_modifications = int(len(onset_times) * 0.3)  # Modify 30% of detected onsets
mod_indices = random.sample(range(len(onset_times)), num_modifications)

# Stretch/squeeze factors (randomized)
stretch_factors = {i: random.uniform(0.2, 2) for i in mod_indices}

for i in mod_indices:
  stretch_factors[i-2] = stretch_factors[i]
  stretch_factors[i-1] = stretch_factors[i]
  stretch_factors[i+1] = stretch_factors[i]
  stretch_factors[i+2] = stretch_factors[i]

# Processed audio storage
y_modified = np.array([])

prev_frame = 0
for i, onset in enumerate(onset_frames):
    frame_start = onset * 512  # Convert frame index to sample index
    frame_end = onset_frames[i + 1] * 512 if i + 1 < len(onset_frames) else len(y)

    segment = y[frame_start:frame_end]

    # Stretch/squeeze only for selected onsets
    if i in mod_indices:
        factor = stretch_factors[i]
        segment = librosa.effects.time_stretch(segment, rate=factor)
        print("Modified segment at {}s with factor {}".format(onset_times[i], factor))

    y_modified = np.concatenate((y_modified, segment))

# Recalculate Onsets
onset_frames_final = librosa.onset.onset_detect(y=y_modified, sr=sr)
onset_times_final = librosa.frames_to_time(onset_frames_final, sr=sr)
print("Final {} onsets detected.".format(len(onset_times_final)))

# Save the modified audio as WAV first
output_wav = "./outputs/stretch-squeeze-on-onsets.wav"
sf.write(output_wav, y_modified, sr)

# Convert to MP3 using pydub
# output_mp3 = "output_final.mp3"
# AudioSegment.from_wav(output_wav).export(output_mp3, format="mp3")
# print(f"Saved final MP3 as {output_mp3}")
