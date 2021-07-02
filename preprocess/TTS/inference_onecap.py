# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from Tacotron2.text import text_to_sequence
import models
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import pickle
import sys
import os
import math
import time
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger.autologging import log_hardware, log_args
from torch.utils.data import Dataset
from tqdm import tqdm
import random
# from apex import amp

from waveglow.denoiser import Denoiser

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, default= 'data/input/Demo.pickle',
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', default= 'data/output',  #'
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--tacotron2', type=str, default='models/tacotron2_statedict.pt',
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,default= 'models/waveglow_256channels.pt',
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--amp-run', default = False,
                        help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', default= False,
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')


    return parser


def load_checkpoint(checkpoint_path, model_name):
    assert os.path.isfile(checkpoint_path)

    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    print("Loaded '{}' checkpoint '{}'" .format(model_name, checkpoint_path))
    return model


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, amp_run):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, to_cuda=True)

    if checkpoint is not None:
        
        state_dict = torch.load(checkpoint)['state_dict']
        # if checkpoint_from_distributed(state_dict):
        #     state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if amp_run:
        model, _ = amp.initialize(model, [], opt_level="O3")

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if torch.cuda.is_available():
        text_padded = torch.autograd.Variable(text_padded).cuda().long()
        input_lengths = torch.autograd.Variable(input_lengths).cuda().long()
    else:
        text_padded = torch.autograd.Variable(text_padded).long()
        input_lengths = torch.autograd.Variable(input_lengths).long()

    return text_padded, input_lengths


class MeasureTime():
    def __init__(self, measurements, key):
        self.measurements = measurements
        self.key = key

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def collate_fn(batch):   
    tests,keys = zip(*batch)
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in tests]),
    
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(tests), max_input_len)
    index = torch.LongTensor(keys)[ids_sorted_decreasing]
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = tests[ids_sorted_decreasing[i]]
        
        text_padded[i, :text.size(0)] = text
       
    return text_padded, input_lengths, index

class dataloader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,args):
        with open(args.input,'rb') as f:
            # the pickle file is a file of list of strings: ['string1.', 'string2.', ...]
            total = pickle.load(f)
            # randomly choose only one of the captions
            self.texts = [random.choice(total)] #[:1100]
        print("===============================len of texts",len(self.texts))
        #print(f"loaded texts: {self.texts}")
        self.keys = np.arange(start=0,stop=len(total),step=1)
        # with open('save_audiofiles.pickle','rb') as f1:
        #     self.save_filname = pickle.laod(f1)
        

           
    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, ['english_cleaners']))
        return text_norm

    def __getitem__(self, index):
        
        # Rindx = 0-index-1
        # return self.get_text(self.texts[Rindx]), self.keys[Rindx]
        return self.get_text(self.texts[index]), self.keys[index]

    def __len__(self):
        return len(self.texts)


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # LOGGER.set_model_name("Tacotron2_PyT")
    # LOGGER.set_backends([
    #     dllg.StdOutBackend(log_file=None,
    #                        logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
    #     dllg.JsonBackend(log_file=args.log_file,
    #                      logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    # ])
    # LOGGER.register_metric("tacotron2_items_per_sec", metric_scope=dllg.TRAIN_ITER_SCOPE)
    # LOGGER.register_metric("tacotron2_latency", metric_scope=dllg.TRAIN_ITER_SCOPE)
    # LOGGER.register_metric("waveglow_items_per_sec", metric_scope=dllg.TRAIN_ITER_SCOPE)
    # LOGGER.register_metric("waveglow_latency", metric_scope=dllg.TRAIN_ITER_SCOPE)
    # LOGGER.register_metric("latency", metric_scope=dllg.TRAIN_ITER_SCOPE)

    # log_hardware()
    # log_args(args)

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.amp_run)

    waveglow = torch.load(args.waveglow)['model']
    waveglow.cuda().eval()
    # waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
    #                                 args.amp_run)
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow).cuda()

    data_loader = torch.utils.data.DataLoader(dataloader(args), 5, shuffle=False, collate_fn = collate_fn)  

    measurements = {}
    img_num = 0
    k = 0
    for i, data in tqdm(enumerate(data_loader)):
        try: 
            new_num = math.ceil((i+1)/2)             
            sequences_padded, input_lengths, keys = data
            if torch.cuda.is_available():
                sequences_padded = torch.autograd.Variable(sequences_padded).cuda().long()
                input_lengths = torch.autograd.Variable(input_lengths).cuda().long()
            else:
                sequences_padded = torch.autograd.Variable(sequences_padded).long()
                input_lengths = torch.autograd.Variable(input_lengths).long()



            with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
                _, mel, _, _, mel_lengths = tacotron2.infer(sequences_padded, input_lengths)

            with torch.no_grad(), MeasureTime(measurements, "waveglow_time"):
                audios = waveglow.infer(mel, sigma=args.sigma_infer)
                audios = audios.float()
                audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)




            # tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']
            # waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']

            # LOGGER.log(key="tacotron2_items_per_sec", value=tacotron2_infer_perf)
            # LOGGER.log(key="tacotron2_latency", value=measurements['tacotron2_time'])
            # LOGGER.log(key="waveglow_items_per_sec", value=waveglow_infer_perf)
            # LOGGER.log(key="waveglow_latency", value=measurements['waveglow_time'])
            # LOGGER.log(key="latency", value=(measurements['tacotron2_time']+
            #                                  measurements['waveglow_time']))

            for j, audio in enumerate(audios):
                k+=1
                key = keys[j]
                print(f"{j}: key - {key}")
                audio = audio[:mel_lengths[j]*args.stft_hop_length]
                audio = audio/torch.max(torch.abs(audio))
                # audio_path = args.output + "/audio_"+str(j)+'-'+str(i)+".wav"
                audio_dir = args.output
                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir, exist_ok=False)
                audio_path = f"{str(key)}.wav"
                save_path = os.path.join(audio_dir, audio_path)
                print(f'saving to path: {save_path}')
                write(save_path, args.sampling_rate, audio.cpu().numpy())

                info = 'saved the %i-th audios\n'%(k)       
              
        except Exception as e:
            print(f'something went wrong? {e}')
            pass

        # LOGGER.iteration_stop()
        # LOGGER.finish()

if __name__ == '__main__':
    main()