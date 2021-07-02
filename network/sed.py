import torch
import torch.nn as nn
import torch.nn.functional as F


class SED(nn.Module):
    """
    Speech encoder network
    """

    def __init__(self):
        super(SED, self).__init__()

        self.conv1 = nn.Conv1d(40, 50, kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.Conv1d(50, 128, kernel_size=6, stride=2, padding=0)

        self.bnorm1 = nn.BatchNorm1d(50)
        self.bnorm2 = nn.BatchNorm1d(128)

        self.gru = torch.nn.GRU(128, 512, num_layers=2, bidirectional=True, batch_first=True, dropout=0.0)
        self.att = multi_attention(in_size=1024, hidden_size=128, n_heads=1)

    def forward(self, audio_input, length):
        # print(f"x.shape: {audio_input.shape}")
        audio_input = audio_input.transpose(2, 1)
        # print(f"x.shape: {audio_input.shape}")
        x = self.conv1(audio_input)
        x = self.bnorm1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)

        #l = [int((y-(self.Conv1.kernel_size[0]-self.Conv1.stride[0]))/self.Conv1.stride[0]) for y in l]
        #l = [int((y-(self.Conv2.kernel_size[0]-self.Conv2.stride[0]))/self.Conv2.stride[0]) for y in l]

        length = [int((l - (self.conv1.kernel_size[0] - self.conv1.stride[0])) / self.conv1.stride[0]) for l in length]
        length = [int((l - (self.conv2.kernel_size[0] - self.conv2.stride[0])) / self.conv2.stride[0]) for l in length]
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2, 1), length, batch_first=True)

        x, hn = self.gru(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.att(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x


# Attention module from https://github.com/xinshengwang/S2IGAN

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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(torch.tan(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x


def main():
    # testing SED
    audio_model = SED()
    print(audio_model)
    input_length = torch.tensor([609., 253.])
    input_audio = torch.rand((2, 609, 40), dtype=torch.float32)
    print(input_length)
    print(input_audio.shape)
    print(input_audio[0])
    print(input_audio[1])

    output_audio = audio_model(input_audio, input_length)
    print(output_audio)


if __name__ == '__main__':
    main()
