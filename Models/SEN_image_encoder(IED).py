"""
The image encoder (IED):

The image encoder (IED) adopts the Inception-v3 pretrained
on ImageNet to extract visual features. On top of
it, a linear layer (FC) is employed to convert the visual feature to
a common space of visual and speech embeddings. The input
size of this linear layer is 2048 and the output size in the
common embedding space is 1024. As a result, we obtain an
image embedding xi which is a 1024-d vector from the image
encoder.[manuscript]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models as imagemodels
#import torch.utils.model_zoo as model_zoo
from torchvision import models
from cfg.Pretrain.config import cfg

## Pre-trained Inception V3 Model

class Inception_V3_Model(nn.Module):

    def __init__(self):
        super(Inception_V3_Model, self).__init__()

        model = models.inception_v3()
        pretrained_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(pretrained_url))
        for param in model.parameters():
            param.requires_grad = False

        print("Loading pre-trained model from ", pretrained_url)
        # print(model

        self.define_inception_module(model)



    def define_inception_module(self, model):
        ## define our model architecture with the pretrained model

        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c


    def forward(self, x):
        features = None

        # fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        x = x.mean(dim=(2, 3)) # 2048

        return x  # or return nn.functional.normalize(x, p=2, dim=1) #cnn_code  #1024




## Sigle layer FC layer 

class Linear_Encoder(nn.Module):
    def __init__(self):
        super(Linear_Encoder, self).__init__()

        self.Linear = nn.Linear(cfg.IMGF.input_dim, cfg.IMGF.embedding_dim)

    def init_trainable_weights(self):
        initrange = 0.1
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        
        if len(input.shape) == 3:
            input = input.squeeze(1)
            
        x = self.Linear(input)
        return nn.functional.normalize(x, p=2, dim=1)