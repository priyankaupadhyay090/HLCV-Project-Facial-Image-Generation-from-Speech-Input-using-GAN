#!/usr/bin/env bash
cd TTS/
for i in {0..9}
do
  python inference.py --input ../mmca/captions_pickles/$i.pickle --output ../mmca/audio/wav/$i/ --tacotron2 saved_models/tacotron2_statedict.pt --waveglow saved_models/waveglow_256channels_universal_v5.pt
done

#convert wav to npy and mel
cd ../
python wavaudio_to_npy.py
python wavaudio_to_mel.py
