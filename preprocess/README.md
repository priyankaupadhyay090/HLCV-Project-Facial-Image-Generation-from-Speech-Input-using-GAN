# TTS
1. Generate .wav files from text strings
2. text or caption should be processed into a list of strings, tokenized and all lowercase.
3. .wav audio files need to be further processed into mel spectrogram files

## Tacotron2 pre-processing
1. clone [https://github.com/xinshengwang/Tacotron2_batch_inference](https://github.com/xinshengwang/Tacotron2_batch_inference) repo
2. Fix the bug in the waveglow dependency as follows:
	1. clone the [original Tacotron2 repo](https://github.com/NVIDIA/tacotron2)
	2. CD into this repo: `cd tacotron2`
	3. Initialize and update the submodule: `git submodule init; git submodule update`
	4. copy the `glow.py` and `glow_old.py` from this repo into the `Tacotron2_batch_inference` directory to replace the version there
3. Before running the inference to generate the .wav files, need to download the pre-trained Tacotron2 and Waveglow checkpoints
4. Checkpoints can be downloaded from the [oritinal Tacotron2 repo](https://github.com/NVIDIA/tacotron2)
5. cd back to `Tacotron2_batch_inference` directory
6. mkdir data/output; this will be where the .wav files are saved
7. run `python inference.py --tacotron2 <path to where tacotron2 chkpt is saved> --waveglow <path to saved waveglow chkpt>

## .wav audio files to Mel
1. run `python wavaudio_to_npy.py` to read wav to np arrays and save them as .npy files
2. these files are saved under `audio_npy`
3. run `python wavaudio_to_mel.py` to convert the .npy files in `audio_npy` into mel spectogram files and save them in 
   `audio_mel` dir as .npy files with the same structure and filenames
   
4. the .npy files in `audio_mel` have identical names as in `audio_npy`

# Data Files

The data files we work with can be downloaded from [this Google Drive](https://drive.google.com/drive/folders/1e2PfNi5YuQYzrzpPeOQ6DTJDht5FGoJg?usp=sharing).
This drive also contains the original downloaded **Multi-Modal-Celeb-A Dataset**, which can also be downloaded from: 
[Multi-Modal-Celeb-A Dataset Source](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset).

For our experiments, you only need to download the following files:
* image files
* mel spectrograms of the text captions files (~2.5GB zip file). Please note that the version provided in our [Google Drive](https://drive.google.com/drive/folders/1e2PfNi5YuQYzrzpPeOQ6DTJDht5FGoJg?usp=sharing) only contains the mel spectrograms of 1 randomly selected caption (from 10 captions) per image file.
* test and train partition filenames

If you also want the .wav audio files of the captions, they can be downloaded from this other [Google Drive](https://drive.google.com/drive/folders/1pCE2M2pVDnZLNlUiLG6340QSEAsAR3lQ?usp=sharing). (~15GB zip file) Please note that this version only contains 1 randomly selected caption (from 10 captions) per image file.

**10 samples** of the images, .wav, and mel spectrogram files can also be downloaded directly from the git repo.

Please organize the data directory as follows:

```angular2html
mmca:
	|-images: <put all image files here>
	|
	|-audio
		|-mel: <put all mel spectogram files here>
		|-wav: <all wav files from Tacotron2 inference.py should be here>
	|-test: <filename.pickle file for test partition filenames>
	|-train: <filename.pickle file for train partition filenames>
	|-captions_pickles: <captions for each image from captions_to_pickle.py>
	|-celeba-caption: <original caption files in .txt>
```

To be able to run all the pre-processing scripts, all the files need to be in these directories according to this
organizational structure. For training and running experiments, only the files in `mmca/images`, 
`mmca/audio/mel`, `test`, and `train` need to be there.

The mel spectrogram files can be generated from celeba-caption according to the following steps:

1. `python captions_to_pickle.py`
2. `bash run_inference_celeba.sh`
3. `python wavaudio_to_npy.py`
4. `python wavaudio_to_mel.py`

Steps 2-4 can also be run with: `bash run_inference_wave_to_mel.sh` 

Esimated pre-processing time: **5-6 secs/image** from TTS - Mel. There are **30,000** images.
