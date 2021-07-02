import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
from pathlib import Path
from importlib import reload

from dataset.SpeechImgDataset import SpeechImgDataset, worker_init_fn, pad_collate
from network.ied import Inception_V3_Model, Linear_Encoder
from network.sed import SED
import pretrain

PARENT_DIR = Path(__file__).resolve().parent

# parameters from cfg or cfg file in S2IGAN
MANUAL_SEED = 200
TRAIN_MODE = 'co-train'  # 'extraction' during feature extraction
CAPTIONS_PER_IMAGE = 10  # number of captions per image
TRAIN_FLAG = True  # change to False if NOT training
TRAIN_BATCH_SIZE = 2
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 0  # 8 if DEVICE == 'cuda' else 2  # set this to 0 if [W ParallelNative.cpp:212] Warning Error

# sample pickle files for filenames: ['0', '1', '2', etc.]
FILENAMES = 'sample_filenames.pickle'

print(f"Testing Constant Variables:\n"
      f"PARENT_DIR: {PARENT_DIR}\n"
      f"TRAIN_MODE: {TRAIN_MODE}\n"
      f"TRAIN_FLAG: {TRAIN_FLAG}\n"
      f"MANUAL_SEED: {MANUAL_SEED}\n"
      f"CAPTIONS_PER_IMAGE: {CAPTIONS_PER_IMAGE}\n"
      f"TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}\n"
      f"WORKERS: {WORKERS}\n"
      f"DATA_DIR: {DATA_DIR}\n"
      f"IMG_SIZE: {IMG_SIZE}\n"
      f"DEVICE: {DEVICE}")

# random seed
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed_all(MANUAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

split_dir, bshuffle = 'train', True
if not TRAIN_FLAG:
    # bshuffle = False
    split_dir = 'test'

# Get data loader
imsize = IMG_SIZE
image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

if TRAIN_MODE == 'co-train':
    dataset = SpeechImgDataset(DATA_DIR, 'train',
                               img_size=imsize,
                               transform=image_transform)
    dataset_test = SpeechImgDataset(DATA_DIR, 'test',
                                    img_size=imsize,
                                    transform=image_transform)

    assert dataset

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=TRAIN_BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=WORKERS, collate_fn=pad_collate,
        worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=TRAIN_BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=WORKERS, collate_fn=pad_collate,
        worker_init_fn=worker_init_fn)

# set up models
speech_model = SED()
image_model = Inception_V3_Model()
linear_model = Linear_Encoder()

pretrain.train(train_loader, val_loader, speech_model, image_model, linear_model)

