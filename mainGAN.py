from __future__ import print_function
from main import BATCH_SIZE, MANUAL_SEED
from dataset import SpeechImgDataset
import torch
import torchvision.transforms as transforms
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import sys
import numpy as np
from pathlib import Path

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

MANUAL_SEED = 200
parent_dir = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'
saving_dir = parent_dir / "GAN_output"
BATCH_SIZE = 32
B_CONDITION = True


MODAL = "train"  # | "test"


if __name__ == "__main__":

    if MODAL == "test":
        MANUAL_SEED = 100
    random.seed(MANUAL_SEED)    
    np.random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(MANUAL_SEED)
        torch.cuda.manual_seed_all(MANUAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

    def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
        np.random.seed(args.manualSeed + worker_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())

    output_dir = saving_dir

    split_dir, bshuffle = 'train', True

    if MODAL == "test":        
        bshuffle = False
        split_dir = 'test'

    imsize = 299

    image_transform = transforms.Compose([
        transforms.Resize(int(imsize)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    train_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, split_dir, transform=image_transform)
    #num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                             drop_last=True, shuffle=bshuffle, num_workers=0, worker_init_fn=worker_init_fn)

    if not B_CONDITION:
        from trainer import GANTrainer as trainer
    else:
        from trainer import condGANTrainer as trainer

    algo = trainer(output_dir, dataloader, imsize)

    start_t = time.time()
    if MODAL == "train":
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)