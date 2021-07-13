import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
from pathlib import Path
import network
import pretrain



from dataset import SpeechImgDataset

PARENT_DIR = Path().resolve()
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'

train_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'train')
test_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'test')

train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32,
            drop_last=True, shuffle=True, num_workers=8, collate_fn=SpeechImgDataset.pad_collate)

test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=32,
            drop_last=False, shuffle=False, num_workers=8, collate_fn=SpeechImgDataset.pad_collate)


speech_model = network.SED()
image_model = network.Inception_V3_Model()
linear_model = network.Linear_Encoder()

pretrain.train(train_loader, test_loader, speech_model, image_model, linear_model)
print("model trained")
torch.save(speech_model.state_dict(), "state_dict_speech_model.pt")