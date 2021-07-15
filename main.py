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
import logging
from datetime import datetime
import random

PARENT_DIR = Path().resolve()
DATA_DIR = PARENT_DIR / 'preprocess' / 'mmca'
LOGGING_DIR = PARENT_DIR / 'logging'
SAVING_DIR = PARENT_DIR / "SEN_saved_models"
MANUAL_SEED = 200

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


# set seed
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed_all(MANUAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(MANUAL_SEED + worker_id)


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    filename=logging_filename,
                    level=logging.INFO)


train_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'train')
test_data = SpeechImgDataset.SpeechImgDataset(DATA_DIR, 'test')

train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32,
            drop_last=True, shuffle=True, num_workers=8,
            collate_fn=SpeechImgDataset.pad_collate, worker_init_fn=worker_init_fn)

test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=32,
            drop_last=False, shuffle=False, num_workers=8,
            collate_fn=SpeechImgDataset.pad_collate, worker_init_fn=worker_init_fn)


speech_model = network.SED()
image_model = network.Inception_V3_Model()
linear_model = network.Linear_Encoder()

pretrain.train(train_loader, test_loader, speech_model, image_model, linear_model)
print("model trained")
saved_model_filename = SAVING_DIR / "last_epoch_state_dict_speech_encoder.pt"
logger.info(f"Training finished, saving last epoch speech model to {saved_model_filename}")
torch.save(speech_model.state_dict(), saved_model_filename)
logger.info(f"====================FINISHED====================")
