"""
This script is adapted from the Audio_to_npy.py script in https://github.com/xinshengwang/S2IGAN.
Read in all the .wav audio files in a specified directory and save the list of np.arrays as .npy files
"""
import numpy as np
import librosa
import os
from tqdm import tqdm

path = 'mmca/audio/wav/'
img_names = os.listdir(path)
save_root = 'mmca/audio/audio_npy'
#for clss_name in sorted(clss_names):
#    print(f"clss_name: {clss_name}")
#    clss_path = os.path.join(path, clss_name)
#    try:
#        img_names = os.listdir(clss_path)
#        print(f"img_names: {img_names}")
#    except NotADirectoryError:
#        img_names = [clss_path]
for img_name in tqdm(sorted(img_names), desc="images"):
    img_path = os.path.join(path, img_name)
    # print(f"img_path: {img_path}")
    try:
        audio_names = os.listdir(img_path)
    except NotADirectoryError or FileNotFoundError:
        audio_names = [img_name]
    audio = []
    for audio_name in sorted(audio_names):
        audio_path = os.path.join(img_path, audio_name)
        if os.path.exists(audio_path):
            y, sr = librosa.load(audio_path)
        else:
            y, sr = librosa.load(audio_name)
        audio.append(y)
    save_path = save_root + '/' + img_name

    if save_path.endswith('.wav'):
        save_path = save_path.replace('.', '_')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_path.endswith('_wav'):
        save_name = save_path + '/' + save_path.split('/')[-1] + '.npy'
    else:
        save_name = save_path + '/' + img_name + '.npy'
    np.save(save_name, audio)
    print(f"{img_name} wav files to npy processed."
          f"files saved at: {save_name}")
