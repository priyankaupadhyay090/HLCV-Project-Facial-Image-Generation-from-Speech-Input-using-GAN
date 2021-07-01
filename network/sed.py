import torch
import torch.nn as nn
import torch.nn.functional as F

class SED(nn.Module):

    """
    Speech encoder network
    """
    def __init__(self):
        super(SED, self).__init__()

        self.conv1 = nn.Conv1d(40, 64, kernel_size = 6, stride = 2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size = 6, stride = 2)

        self.bnorm1 = nn.BatchNorm1d(64)
        self.bnorm2 = nn.BatchNorm1d(128)

        self.gru = torch.nn.GRU(128, 512, num_layers = 2, bidirectional = True, batch_first = True)
        self.att = multi_attention(in_size = 1024, hidden_size = 128, n_heads = 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)

        #l = int((l-(self.conv1.kernel_size[0]-self.conv1.stride[0]))/self.conv1.stride[0])
        #l = int((l-(self.conv2.kernel_size[0]-self.conv2.stride[0]))/self.conv2.stride[0])
        #x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)

        x, hn = self.gru(x.transpose(2,1))
        x = self.att(x)

        return x

#Attention module from https://github.com/xinshengwang/S2IGAN

class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)

class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(torch.tan(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x 