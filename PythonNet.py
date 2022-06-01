import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utils import Bar, AverageMeter


class ResBlock(nn.Module):
    def __init__(self, nf):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, stride=1)

        self.bn1 = nn.BatchNorm2d(nf)
        self.bn2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(residual + x)
        return x


class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        self.ngpu = args.ngpu
        self.conv1 = nn.Conv2d(2, 75, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(75)

        self.res_layers = nn.ModuleList([ResBlock(75) for _ in range(5)])
        self.conv3 = nn.Conv2d(75, 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2)
        self.flatten = nn.Flatten()
        self.dense_pi = nn.Linear(190, args.n_out, bias=True)
        # self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, training=False):
        x = F.leaky_relu(self.bn1(self.conv1(x)))

        for i in range(5):
            x = self.res_layers[i](x)

        pi = F.leaky_relu(self.bn3(self.conv3(x)))
        pi = self.flatten(pi)
        pi = self.dense_pi(pi)

        if training:
            print(np.shape(pi))

        # pi = self.softmax(pi)

        return pi


class ResNN2D():
    def __init__(self, args):
        self.args = args
        self.nn = NN(args)
        self.target_nn = NN(args)

    def predict(self, inputToModel):
        inputToModel = torch.FloatTensor(inputToModel.astype(np.float64), device=self.args.device)
        with torch.no_grad():
            inputToModel = Variable(inputToModel)

        self.nn.eval()
        preds = self.nn(inputToModel)
        return preds

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nn.state_dict(),
                    }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
            return
        if torch.cuda.is_available():
            checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.nn.load_state_dict(checkpoint['state_dict'])