import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    def __init__(self, args):
        super(SimpleNN, self).__init__()
        self.ngpu = args.ngpu
        self.dense1 = nn.Linear(args.n_in, 64, bias=False)
        self.dense2 = nn.Linear(64, 32, bias=False)
        self.dense3 = nn.Linear(32, args.n_out, bias=False)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        return x
