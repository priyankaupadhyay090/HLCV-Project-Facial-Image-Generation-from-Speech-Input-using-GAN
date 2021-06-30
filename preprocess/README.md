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
