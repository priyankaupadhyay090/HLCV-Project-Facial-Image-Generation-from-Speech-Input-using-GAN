import os
import torch
from torch._C import device
import torch.nn as nn
from sen_utils import pre_utils
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
import logging
import time
from sen_utils import pre_utils


logger = logging.getLogger(__name__)
parent_dir = Path(__file__).resolve().parent
saving_dir = parent_dir / "SEN_saved_models"


def train(train_loader, test_loader, speech_encoder, image_encoder, linear_encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"device: {device}, n_gpu: {torch.cuda.device_count()}")
    torch.set_grad_enabled(True)

    """ model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    """
    # set up tracking eval metrics progress
    batch_time = pre_utils.AverageMeter()
    data_time = pre_utils.AverageMeter()
    loss_meter = pre_utils.AverageMeter()

    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()

    if torch.cuda.device_count() > 1:
        if not isinstance(speech_encoder, torch.nn.DataParallel):
            speech_encoder = nn.DataParallel(speech_encoder)

        if not isinstance(image_encoder, torch.nn.DataParallel):
            image_encoder = nn.DataParallel(image_encoder)

        if not isinstance(linear_encoder, torch.nn.DataParallel):
            linear_encoder = nn.DataParallel(linear_encoder)

    if epoch != 0:
        speech_encoder.load_state_dict(torch.load(saving_dir / f"best_speech_encoder_{epoch}.pt"))
        linear_encoder.load_state_dict(torch.load(saving_dir / f"best_linear_encoder_{epoch}.pt"))

    speech_encoder = speech_encoder.to(device)
    image_encoder = image_encoder.to(device)
    linear_encoder = linear_encoder.to(device)

    # set up optimizer
    audio_trainables = [p for p in speech_encoder.parameters() if p.requires_grad]
    image_trainables = [p for p in linear_encoder.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables

    max_epoch = 120
    lr = 1e-4
    lr_decay = 50
    bs = 128

    logger.info(f"Max Epoch: {max_epoch}, LR: {lr}, batch size: {bs}\n")

    optimizer = torch.optim.Adam(trainables, lr=lr,
                                 weight_decay=1e-3,
                                 betas=(0.95, 0.999))

    print("start training")
    print(f"current steps: {global_step} | epochs: {epoch}")
    logger.info(f"==============TRAINING STARTED=================\n")

    image_encoder.eval()

    while epoch <= max_epoch:
        epoch += 1
        adjust_learning_rate(lr, lr_decay, optimizer, epoch)
        end_time = time.time()

        speech_encoder.train()
        linear_encoder.train()

        for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(tqdm(train_loader,
                                                                                              desc='training',
                                                                                              total=len(train_loader))):
            # updating data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.float().to(device)
            label = label.long().to(device)
            input_length = input_length.float().to(device)

            image_input = image_input.to(device)
            image_input = image_input.squeeze(1)

            optimizer.zero_grad()

            image_output = linear_encoder(image_encoder(image_input))
            audio_output = speech_encoder(audio_input.permute(0, 2, 1), input_length)

            loss = 0

            loss_xy, loss_yx = pre_utils.batch_loss(image_output, audio_output, bs=len(audio_input), class_ids=label)
            loss_batch = loss_xy + loss_yx
            loss += loss_batch * 1.0  # cfg.Loss.gamma_batch
            #loss = batch_loss(image_output, audio_output, bs=len(audio_input), class_ids=label)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if i % 5 == 0:
                print(f"epoch: {epoch} - iteration: {i} | loss: {loss}")
                logger.debug(f"epoch: {epoch} - iteration: {i} | loss: {loss}")

            end_time = time.time()
            global_step += 1

        if epoch % 5 == 0:
            recalls = validation(speech_encoder, linear_encoder, image_encoder, test_loader)

            A_r10 = recalls['A_r10']
            I_r10 = recalls['I_r10']
            A_r5 = recalls['A_r5']
            I_r5 = recalls['I_r5']
            A_r1 = recalls['A_r1']
            I_r1 = recalls['I_r1']
            medr_I2A = recalls['medr_I2A']
            medr_A2I = recalls['medr_A2I']
            avg_acc = (A_r10 + I_r10) / 2

            print(A_r10)
            print(I_r10)
            print(medr_A2I)
            print(medr_I2A)
            print(avg_acc)

            info = ' Epoch: [{0}] Loss: {loss:.4f} | \
                    *Audio:R@1 {A_r1:.4f} R@5 {A_r5:.4f} R@10 {A_r10:.4f} medr {A_m:.4f}| *Image R@1 {I_r1:.4f} R@5 {I_r5:.4f} R@10 {I_r10:.4f} \
                   medr {I_m:.4f} \n'.format(epoch, loss=loss, A_r1=A_r1, A_r5=A_r5, A_r10=A_r10, A_m=medr_A2I, I_r1=I_r1,
                                             I_r5=I_r5, I_r10=I_r10, I_m=medr_I2A)
            print(info)
            logger.info(f"current training eval status: \n{info}\n")
            logger.info(f"epoch: {epoch} | loss: {loss} | avg_accuracy: {avg_acc}\n")

            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc

                try:
                    torch.save(speech_encoder.state_dict(), saving_dir / f"best_speech_encoder_{best_epoch}.pt")
                    torch.save(linear_encoder.state_dict(), saving_dir / f"best_linear_encoder_{best_epoch}.pt")
                    torch.save(optimizer.state_dict(), saving_dir / f"optim_state_{best_epoch}.pt")
                except FileNotFoundError:
                    saving_dir.mkdir(parents=True, exist_ok=False)
                    print(f"Creating {saving_dir} to save best models...")
                    torch.save(speech_encoder.state_dict(), saving_dir / f"best_speech_encoder_{best_epoch}.pt")
                    torch.save(linear_encoder.state_dict(), saving_dir / f"best_linear_encoder_{best_epoch}.pt")
                    torch.save(optimizer.state_dict(), saving_dir / f"optim_state_{best_epoch}.pt")
                logger.info(f"best models and optimizer state saved in {saving_dir}")
        # implement accuracy

        # implement best model picking


# implement validation


def validation(speech_encoder, linear_encoder, image_encoder, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = pre_utils.AverageMeter()

    # in case of multiple gpus
    if torch.cuda.device_count() > 1:
        if not isinstance(speech_encoder, torch.nn.DataParallel):
            speech_encoder = nn.DataParallel(speech_encoder)
        if not isinstance(image_encoder, torch.nn.DataParallel):
            image_encoder = nn.DataParallel(image_encoder)
        if not isinstance(linear_encoder,torch.nn.DataParallel):
            linear_encoder = nn.DataParallel(linear_encoder)

    speech_encoder = speech_encoder.to(device)
    linear_encoder = linear_encoder.to(device)
    image_encoder = image_encoder.to(device)

    # eval mode
    speech_encoder.eval()
    linear_encoder.eval()
    image_encoder.eval()

    end = time.time()
    I_embeddings = []
    A_embeddings = []

    I_class_ids = []
    A_class_ids = []

    with torch.no_grad():
        for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(tqdm(val_loader,
                                                                                              desc='validating',
                                                                                              total=len(val_loader))):
            image_input, inverse = torch.unique(image_input, sorted=False, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(image_input.size(0)).scatter_(0, inverse, perm)

            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)

            image_feat = image_encoder(image_input)
            image_output = linear_encoder(image_feat)
            audio_output = speech_encoder(audio_input.permute(0, 2, 1), input_length)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)
            I_class_ids.append(cls_id[perm])
            A_class_ids.append(cls_id)

            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)

        I_ids = torch.cat(I_class_ids)
        A_ids = torch.cat(A_class_ids)

        image_output, inverse = torch.unique(image_output, sorted=False, return_inverse=True, dim=0)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(image_output.size(0)).scatter_(0, inverse, perm)
        I_ids = I_ids[perm]
        recalls = retrieval_evaluation_all(image_output, audio_output, I_ids, A_ids)

    return recalls


def retrieval_evaluation_all(image_output, audio_output, I_id, A_id):
    img_f = normalizeFeature(image_output)
    aud_f = normalizeFeature(audio_output)
    S = img_f.mm(aud_f.t())

    # image to audio retrieval

    _, indx_I2A = torch.sort(S, dim=1, descending=True)
    class_sorted_I2A = A_id[indx_I2A]
    Correct_num_I2A_1 = sum(class_sorted_I2A[:, 0] == I_id)
    Correct_num_I2A_5 = ((class_sorted_I2A[:, :5] == I_id.unsqueeze(-1).repeat(1, 5)).sum(1) != 0).sum()
    Correct_num_I2A_10 = ((class_sorted_I2A[:, :10] == I_id.unsqueeze(-1).repeat(1, 10)).sum(1) != 0).sum()

    Rank1_I2A = Correct_num_I2A_1 * 1.0 / img_f.shape[0]
    Rank5_I2A = Correct_num_I2A_5 * 1.0 / img_f.shape[0]
    Rank10_I2A = Correct_num_I2A_10 * 1.0 / img_f.shape[0]

    Rank_I2A = torch.nonzero(class_sorted_I2A == I_id.unsqueeze(-1))
    kr, inverse = torch.unique(Rank_I2A[:, 0], sorted=False, return_inverse=True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(kr.size(0)).scatter_(0, inverse, perm)

    medr_I2A = Rank_I2A[perm][:, 1].median()

    # for audio to image retrieval

    S_T = S.T
    _, indx_A2I = torch.sort(S_T, dim=1, descending=True)
    class_sorted_A2I = I_id[indx_A2I]
    Correct_num_A2I_1 = sum(class_sorted_A2I[:, 0] == A_id)
    Correct_num_A2I_5 = ((class_sorted_A2I[:, :5] == A_id.unsqueeze(-1).repeat(1, 5)).sum(1) != 0).sum()
    Correct_num_A2I_10 = ((class_sorted_A2I[:, :10] == A_id.unsqueeze(-1).repeat(1, 10)).sum(1) != 0).sum()

    Rank1_A2I_1 = Correct_num_A2I_1 * 1.0 / aud_f.shape[0]
    Rank1_A2I_5 = Correct_num_A2I_5 * 1.0 / aud_f.shape[0]
    Rank1_A2I_10 = Correct_num_A2I_10 * 1.0 / aud_f.shape[0]

    Rank_A2I = torch.nonzero(class_sorted_A2I == A_id.unsqueeze(-1))
    kr, inverse = torch.unique(Rank_A2I[:, 0], sorted=False, return_inverse=True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(kr.size(0)).scatter_(0, inverse, perm)
    medr_A2I = Rank_A2I[perm][:, 1].median()

    recalls = {'A_r1': Rank1_A2I_1, 'A_r5': Rank1_A2I_5, 'A_r10': Rank1_A2I_10,
               'I_r1': Rank1_I2A, 'I_r5': Rank5_I2A, 'I_r10': Rank10_I2A,
               'medr_I2A': medr_I2A, 'medr_A2I': medr_A2I}
    return recalls


def normalizeFeature(x):
    x = x + 1e-10  # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x ** 2, axis=1) ** 0.5  # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat


def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def batch_loss(image_output, audio_output, bs, class_ids, eps=1e-8):
    """  labels = Variable(torch.LongTensor(range(bs)))
    labels = labels.cuda()  

    masks = []
    class_ids = class_ids.detach().cpu().numpy()
    for i in range(bs):
        mask = (class_ids == class_ids[i]).astype(np.uint8)
        mask[i] = 0
        masks.append(mask.reshape((1, -1)))
    masks = np.concatenate(masks, 0)
    masks = torch.ByteTensor(masks)
    masks = masks.to(torch.bool)
    masks = masks.cuda()

    if image_output.dim() == 2:
        image_output = image_output.unsqueeze(0)
        audio_output = audio_output.unsqueeze(0)
    
    image_norm = torch.norm(image_output, 2, dim=2, keepdim=True)
    audio_norm = torch.norm(audio_output, 2, dim=2, keepdim=True)

    cos_sim = torch.bmm(image_output, audio_output.transpose(1, 2))
    norm_batch = torch.bmm(image_norm, audio_norm.transpose(1, 2))
    cos_sim = cos_sim/norm_batch.clamp(min=eps) * 10

    cos_sim_xy = cos_sim.squeeze(0)
    cos_sim_xy.data.masked_fill_(masks, -float('inf'))
    cos_sim_yx = cos_sim_xy.transpose(0, 1)
 

    loss_xy = nn.CrossEntropyLoss()(cos_sim_xy, labels)
    loss_yx = nn.CrossEntropyLoss()(cos_sim_yx, labels)
    
    return loss_xy, loss_yx
    """
    # Alternative loss
    size = image_output.size(0)

    similarities = torch.mm(image_output, audio_output.t())  # 30x30 similarity matrix
    diagonal = similarities.diag().view(image_output.size(0), 1)

    d1 = diagonal.expand_as(similarities)
    d2 = diagonal.t().expand_as(similarities)

    margin = 0.2
    max_violation = True

    # audio cost
    cost_audio = (margin + similarities - d1).clamp(min=0)
    # image cost
    cost_image = (margin + similarities - d2).clamp(min=0)

    mask = torch.eye(similarities.size(0)) > .5

    I = Variable(mask)

    if torch.cuda.is_available():
        I = I.cuda()
    cost_audio = cost_audio.masked_fill_(I, 0)
    cost_image = cost_image.masked_fill_(I, 0)

    if max_violation:
        cost_audio = cost_audio.max(1)[0]
        cost_image = cost_image.max(0)[0]

    return cost_audio.sum() + cost_image.sum()

    """ I = torch.eye(size).cuda()

    cost_1 = torch.clamp(.2 - similarities + correct_sims, min = 0)
    cost_1 = ((1 - I) * cost_1).sort(0)[0][-size:, :]
    cost_2 = torch.clamp(.2 - similarities + correct_sims.view(-1, 1), min = 0)
    cost_2 = ((1 - I) * cost_2).sort(1)[0][:, -size:] 

    cost = cost_1 + cost_2.t()

    return cost.mean()"""
