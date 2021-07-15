import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
from pathlib import Path

from torchvision import transforms
import network
import pretrain
from dataset import SpeechImgDataset
import logging
from datetime import datetime

PARENT_DIR = Path().resolve()
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'
LOGGING_DIR = PARENT_DIR / 'logging'
SAVING_DIR = PARENT_DIR / "saved_models"
MODAL = 'training'

if not SAVING_DIR.exists():
    SAVING_DIR.mkdir(parents=True, exist_ok=True)

# set up logging
try:
    LOGGING_DIR.mkdir(parents=True, exist_ok=False)
    print(f"Making {LOGGING_DIR} for saving log files")
except FileExistsError:
    print(f"Logging directory: {LOGGING_DIR} already exists.")

time_now = datetime.now()
logging_filename = LOGGING_DIR / f"SEN_training_log_{time_now:%d_%m_%Y_%H_%M_%S}.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    filename=logging_filename,
                    level=logging.INFO)


image_transform = transforms.Compose([
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip()])

if MODAL == 'training':

    train_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'train', transform = image_transform)
    test_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'test', transform = image_transform)

    train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=32,
                drop_last=True, shuffle=True, num_workers=0, collate_fn=SpeechImgDataset.pad_collate)

    test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=32,
                drop_last=False, shuffle=False, num_workers=0, collate_fn=SpeechImgDataset.pad_collate)

elif MODAL == 'extraction':

    train_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'train', transform = image_transform)
    test_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'test', transform = image_transform)

    train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=32,
                drop_last=False, shuffle=False, num_workers=0, collate_fn=SpeechImgDataset.pad_collate)

    test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=32,
                drop_last=False, shuffle=False, num_workers=0, collate_fn=SpeechImgDataset.pad_collate)

speech_model = network.SED()
image_model = network.Inception_V3_Model()
linear_model = network.Linear_Encoder()

models = [speech_model, image_model, linear_model]

if MODAL == 'train':
    pretrain.train(train_loader, test_loader, models)
    print("model trained")
if MODAL == 'extraction':
    pretrain.feat_extract_co(speech_model, DATA_DIR)