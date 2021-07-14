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

PARENT_DIR = Path().resolve()
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'

image_transform = transforms.Compose([
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip()])

train_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'train', transform = image_transform,)
test_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'test', transform = image_transform)

train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32,
            drop_last=True, shuffle=True, num_workers=0, collate_fn=SpeechImgDataset.pad_collate)

test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=32,
            drop_last=False, shuffle=False, num_workers=0, collate_fn=SpeechImgDataset.pad_collate)


speech_model = network.SED()
image_model = network.Inception_V3_Model()
linear_model = network.Linear_Encoder()

models = [speech_model, image_model, linear_model]

pretrain.train(train_loader, test_loader, models)
print("model trained")
torch.save(speech_model.state_dict(), "state_dict_speech_model.pt")