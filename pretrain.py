import os 
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def train(train_loader, test_loader, speech_encoder, image_encoder, linear_encoder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    """ model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
 """

    speech_encoder = speech_encoder.to(device)
    image_encoder = image_encoder.to(device)
    linear_encoder = linear_encoder.to(device)

    audio_trainables = [p for p in speech_encoder.parameters() if p.requires_grad]
    image_trainables = [p for p in linear_encoder.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    
    max_epoch = 10
    lr = 0.001
    bs = 32

    optimizer = torch.optim.Adam(trainables, lr=lr,
                                weight_decay=0.001,
                                betas=(0.95, 0.999))

    print("starting training")

    image_encoder.eval()
    
    epoch = 0

    while epoch <= max_epoch:
        epoch +=1
        adjust_learning_rate(lr, 50, optimizer, epoch)

        speech_encoder.train()
        linear_encoder.train()

        for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(train_loader):

            B = audio_input.size(0)
        
            audio_input = audio_input.float().to(device)
            label = label.long().to(device)
            input_length = input_length.float().to(device)

            image_input = image_input.to(device)
            image_input = image_input.squeeze(1)

            optimizer.zero_grad()

            image_output = linear_encoder(image_encoder(image_input))
            audio_output = speech_encoder(audio_input.permute(0,2,1),input_length)

            loss = 0  
            loss_xy, loss_yx = batch_loss(image_output,audio_output, bs = len(audio_input), class_ids = label)
            #loss = batch_loss(image_output,audio_output, bs = len(audio_input), class_ids = label)
            loss += loss_xy + loss_yx

            loss.backward()
            optimizer.step() 

        
        print('epoch = {} | loss = {} '.format(epoch,loss))


            #implement accuracy

            #implement best model picking

#implement validation



def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def batch_loss(image_output, audio_output, bs, class_ids, eps=1e-8):
 
    labels = Variable(torch.LongTensor(range(bs)))
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

    #print(cos_sim.shape)
    #print(image_norm.shape)

    loss_xy = nn.CrossEntropyLoss()(cos_sim_xy, labels)
    loss_yx = nn.CrossEntropyLoss()(cos_sim_yx, labels)
    
    return loss_xy, loss_yx

    #Alternative loss
    """ size = image_output.size(0)

    similarities = torch.mm(image_output, audio_output.t()) #30x30 similarity matrix
    correct_sims = torch.diagonal(similarities)

    I = torch.eye(size).cuda()

    cost_1 = torch.clamp(.2 - similarities + correct_sims, min = 0)
    cost_1 = ((1 - I) * cost_1).sort(0)[0][-size:, :]
    cost_2 = torch.clamp(.2 - similarities + correct_sims.view(-1, 1), min = 0)
    cost_2 = ((1 - I) * cost_2).sort(1)[0][:, -size:]

    cost = cost_1 + cost_2.t()

    return cost.mean() """