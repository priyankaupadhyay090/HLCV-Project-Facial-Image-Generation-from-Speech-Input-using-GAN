# Torch Dataset for Speech and Img

To test this module, run `python SpeechImgDataset.py` from this directory.

A sample output from testing val and train Dataloader is saved in
`module_testing_output.txt`

## To import as module

Assuming the main script is in the project's root:
(`HLCV-Project-Facial-Image-Generation-from-Speech-Input-using-GAN`),

`from dataset.SpeechImgDataset import SpeechImgDataset`

## Note: regarding parallel WORKERS

If running on GPU (and not on a MacOS), WORKERS can be set to 8.
CPU: WORKERS == number of threads on the machine.
PyTorch 1.9 (?) has a MacOS Bug and WORKERS need to be set to 0.
