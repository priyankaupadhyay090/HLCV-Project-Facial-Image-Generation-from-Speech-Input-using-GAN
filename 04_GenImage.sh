data_path=data/Flickr8k
save_root=outputs/gan/flickr
cfg_file=cfg/eval_flickr.yml
seed=100

conda init bash
conda activate base
python main.py --data_dir $data_path \
              --save_root $save_root \
			  --cfg $cfg_file\
			  --manualSeed $seed\
			