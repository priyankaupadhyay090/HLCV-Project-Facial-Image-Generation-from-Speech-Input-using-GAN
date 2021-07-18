"""
Dataset Module is adapted from S2IGAN datasets_pre.py, in this repo:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mainGAN import MODAL

from torch.utils.data.dataloader import default_collate
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import sys
import numpy as np
from PIL import Image
import numpy.random as random
import pickle
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
# parameters from cfg or cfg file in S2IGAN
MANUAL_SEED = 200
TRAIN_MODE = 'co-train'  # 'extraction' during feature extraction
CAPTIONS_PER_IMAGE = 1  # number of captions per image
TRAIN_FLAG = True  # change to False if NOT training
TRAIN_BATCH_SIZE = 32
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'
IMG_SIZE = 299
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 0  # 8 if DEVICE == 'cuda' else 2  # set this to 0 if [W ParallelNative.cpp:212] Warning Error

# sample pickle files for filenames: ['0', '1', '2', etc.]
FILENAMES = 'filenames.pickle'


def worker_init_fn(worker_id):
    """
    After creating the workers, each worker has an independent seed that is initialized to the
    curent random seed + the id of the worker

    """
    np.random.seed(MANUAL_SEED + worker_id)


def pad_collate(batch):
    max_input_len = float('-inf')
    batch = list(filter(lambda x: x[1] is not None, batch)) 
    if TRAIN_MODE != 'extraction':
        for elem in batch:
            imgs, captions, cls_id, key, label = elem
            max_input_len = max_input_len if max_input_len > captions.shape[0] else captions.shape[0]

        for i, elem in enumerate(batch):
            imgs, captions, cls_id, key, label = elem
            input_length = captions.shape[0]
            input_dim = captions.shape[1]

            # print('f.shape: ' + str(f.shape))
            feature = np.zeros((max_input_len, input_dim), dtype=np.float64)
            feature[:input_length, :input_dim] = captions

            batch[i] = (imgs, feature, cls_id, key, input_length, label)

        # sort by input_length
        batch.sort(key=lambda x: x[-2], reverse=True)

    #filter None files for not found files
       
    return default_collate(batch)    


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    # img.show()

    """ # mmca dataset doesn't need bbox
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    """

    if transform is not None:
        img = transform(img)

    return normalize(img)

def get_single_img(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    return normalize(img)

def normalizeFeature(x):	
    
    # x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    # feature_norm =  ((x**2).sum())**0.5           #       np.sum(x**2, axis=1)**0.5 # l2-norm
    # feat = x / feature_norm
    x_max = x.max()
    x_min = x.min()
    feat = (x-x_min)/(x_max-x_min)
    return feat

class SpeechImgDataset(data.Dataset):
    def __init__(self, data_dir, split='train', img_size=64, transform=None, target_transform=None):
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = CAPTIONS_PER_IMAGE
        self.img_size = img_size
        self.data_dir = data_dir
        self.bbox = None
        self.filenames = self._load_filenames(data_dir, split)

        # load the class_id for the whole dataset
        self.class_id = self._load_class_id(len(self.filenames))

        # caculate the sequence label for the sampled dataset
        self.number_example = len(self.filenames)

    def _load_class_id(self, total_num):

        class_id = np.arange(total_num)

        return class_id

    def _load_filenames(self, data_dir, split):
        filepath = data_dir / split / FILENAMES  # filenames.pickle | sample_filenames.pickle
        # filepath = f"{data_dir}/{split}/filenames.pickle"
        if filepath.is_file():
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            #print(f"Load filenames from: {filepath} {len(filenames)}")
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        label = cls_id

        if TRAIN_MODE != 'extraction':
            bbox = None
            data_dir = self.data_dir
            # print(f"Data dir: {data_dir}\n")

            img_path_name = data_dir / 'CelebA-HQ-img' / f"{key}.jpg"
            #print(f'Image path: {img_path_name}')
            # img_name = f"{data_dir}/images/{key}.jpg"
            images = get_imgs(img_path_name, self.img_size, bbox, self.transform, normalize=self.norm)


        # audio mel files
        if self.data_dir.name.find('mmca') != -1:
            audio_file = self.data_dir / 'audio' / 'mel_one' / key / f"{key}.npy"
            # audio_file = f"{self.data_dir}/mmca/audio/mel/{key}/{key}.npy"
        else:
            audio_file = self.data_dir / 'mmca' / 'audio' / 'mel_one' / key / f"{key}.npy"
            # audio_file = f"{self.data_dir}/audio/mel/{key}/{key}.npy"
        #print(f"caption audio file: {audio_file}")

        if self.split == 'train':
            audio_ix = random.randint(0, self.embeddings_num)
        else:
            audio_ix = 0
        try:
            audios = np.load(audio_file, allow_pickle=True)
        except:
            audios = None
        #if len(audios.shape) == 2:
           #audios = audios[np.newaxis, :, :]

        # training vs feature extraction
        if TRAIN_MODE != 'extraction':
            if audios is not None:
                captions = audios[audio_ix]
            else:
                captions = audios
        else:
            captions = audios

        if TRAIN_MODE == 'extraction':
            return captions
        else:
            # print('returning images, captions, cls_id, key, label')
            return images, captions, cls_id, key, label

    def __len__(self):
        return len(self.filenames)


class SpeechImgDatasetGAN(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='melspec',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embedding_type = embedding_type

        self.imsize = []
        for i in range(3):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))

        if MODAL == "":
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_embedding(self, data_dir, embedding_type):
          
        if MODAL == "train":
            embedding_filename = PARENT_DIR / "SEN_saved_models" / "extraction" / 'speech_embeddings_train.pickle'
        else:
            embedding_filename = PARENT_DIR / "SEN_saved_models" / "extraction" / 'speech_embeddings_test.pickle'

        if embedding_type != 'Audio_emb':
            with open(data_dir + embedding_filename, 'rb') as f:
                embeddings = pickle.load(f, encoding="bytes")
                # pdb.set_trace()
                embeddings = np.array(embeddings)
                # if len(embeddings.shape)==2:
                #     embeddings = embeddings[:,np.newaxis,:]
        else:
            with open(embedding_filename, 'rb') as f:
                embeddings = pickle.load(f, encoding="bytes")
                embeddings = np.array(embeddings)

            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        data_dir = self.data_dir
        key = self.filenames[index]
        class_id = self.class_id[index]
        self.class_id = np.array(self.class_id)
        same_index = index
        bbox = None
        data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        #img_name = '%s/images/%s.jpg' % (data_dir, key)
        img_path_name = data_dir / 'CelebA-HQ-img' / f"{key}.jpg"
        imgs = get_imgs(img_path_name, self.imsize, bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if self.class_id[index] == self.class_id[wrong_ix]:
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        same_key = self.filenames[same_index]
        wrong_bbox = None
        same_bbox = None
        #wrong_img_name = '%s/images/%s.jpg' % (data_dir, wrong_key)
        #same_image_name = '%s/images/%s.jpg' % (data_dir, same_key)
        wrong_img_name = data_dir / 'CelebA-HQ-img' / f"{wrong_key}.jpg"
        same_image_name = data_dir / 'CelebA-HQ-img' / f"{same_key}.jpg"
        wrong_imgs = get_imgs(wrong_img_name, self.imsize, wrong_bbox, self.transform, normalize=self.norm)
        same_imgs = get_single_img(same_image_name, self.imsize, same_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]

        
        if self.embedding_type =='melspec':
            embedding = normalizeFeature(embedding)
        
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs,same_imgs, embedding, class_id, key  # captions

    def prepair_test_pairs(self, index):
        data_dir = self.data_dir
        key = self.filenames[index]
        bbox = None
        data_dir = self.data_dir

        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]   #[index,:,:]  changed by shawn


        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)

        if self.embedding_type =='melspec':
            embeddings = normalizeFeature(embeddings)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


def main():
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

    # this section may not be neccessary
    split_dir, bshuffle = 'train', True
    if not TRAIN_FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = IMG_SIZE
    image_transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
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

    # Dataloader for classificaiton of single modal
    elif TRAIN_MODE == 'extraction':
        # this is actually not used in S2IGAN at all, so can be deleted later

        dataset = SpeechImgDataset(DATA_DIR, 'train',
                                   img_size=imsize,
                                   transform=image_transform)
        dataset_test = SpeechImgDataset(DATA_DIR, 'test',
                                        img_size=imsize,
                                        transform=image_transform)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=TRAIN_BATCH_SIZE,
            drop_last=False, shuffle=False, num_workers=WORKERS, collate_fn=pad_collate,
            worker_init_fn=worker_init_fn)
        val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=TRAIN_BATCH_SIZE,
            drop_last=False, shuffle=False, num_workers=WORKERS, collate_fn=pad_collate,
            worker_init_fn=worker_init_fn)
    else:
        (f'Invalid option!')
        sys.exit(1)

    # Testing train_loader
    print(f'Testing train_loader')
    N_examples = train_loader.dataset.__len__()
    print(f"N_examples: {N_examples}")
    for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(train_loader):

        B = audio_input.size(0)

        audio_input = audio_input.float().to(DEVICE)
        label = label.long().to(DEVICE)
        input_length = input_length.float().to(DEVICE)

        image_input = image_input.float().to(DEVICE)
        print(f"image_input from data loader: {image_input}\n"
              f"image input shape: {image_input.shape}")
        image_input = image_input.squeeze(1)

        print(f"{i}-th item in the train_loader:\n"
              f"audio_input.size: {B}\n"
              f"audio_input: {audio_input}\n"
              f"label long tensor?: {label}\n"
              f"input length: {input_length}\n"
              f"image_input after squeeze(1): {image_input}\n")

    print(f"Testing val_loader")
    N_examples = val_loader.dataset.__len__()
    print(f"N_examples: {N_examples}")
    for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(val_loader):
        image_input = image_input.to(DEVICE)
        audio_input = audio_input.to(DEVICE)
        print(f"image_input from data loader: {image_input}\n"
              f"image input shape: {image_input.shape}")
        image_input = image_input.squeeze(1)

        audio_input = audio_input.float().to(DEVICE)
        image_input = image_input.float().to(DEVICE)
        input_length = input_length.float().to(DEVICE)

        print(f"{i}-th item in the val_loader:\n"
              f"audio_input.size: {audio_input.size}\n"
              f"audio_input: {audio_input}\n"
              f"label long tensor?: {label}\n"
              f"input length: {input_length}\n"
              f"image_input after squeeze(1): {image_input}\n")


if __name__ == '__main__':
    main()
