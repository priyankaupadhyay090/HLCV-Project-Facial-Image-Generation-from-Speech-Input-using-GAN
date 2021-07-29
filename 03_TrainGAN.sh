data_path=data/Flickr8k
save_root=outputs/gan/flickr
cfg_file=cfg/flickr_3stages.yml
seed=200

conda init bash
conda activate base
python C:/Users/tomas/Documents/DSAI/SS21/HLCV_Test/S2IGAN/main.py --data_dir $data_path \
              --save_root $save_root \
			  --cfg $cfg_file\
			  --manualSeed $seed\
			