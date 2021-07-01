import os
import collections
import glob
import torch
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from os.path import join as pjoin
from torchvision import transforms, datasets
import pickle

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='melspec',
                 base_size=64, transform=None, target_transform=None):
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299,299)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = data_dir
        #split_dir = os.path.join(data_dir, split)
        self.filenames = self.load_filenames(data_dir, split)
        self.image_data = datasets.ImageFolder(data_dir ,  transform = self.norm)
       
    def load_filenames(self, data_dir, split):      
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):

        key = self.filenames[index]  
        cls_id = self.class_id[index]    
        if self.split =='train':
            label = cls_id