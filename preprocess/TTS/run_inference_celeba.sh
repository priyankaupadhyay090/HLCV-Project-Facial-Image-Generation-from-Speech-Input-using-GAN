#!/usr/bin/env bash
for i in {0..29999}
do
  python inference.py --input ../mmca/captions_pickles/$i.pickle --output ../mmca/audio/wav/$i/ --tacotron2 saved_models/tacotron2_statedict.pt --waveglow saved_models/waveglow_256channels_universal_v5.pt
done
