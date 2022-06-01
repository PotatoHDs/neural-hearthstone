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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, training=False):
        if training:
            print(np.shape(x))
        x = F.leaky_relu(self.bn1(self.conv1(x)))

        for i in range(5):
            x = self.res_layers[i](x)

        pi = F.leaky_relu(self.bn3(self.conv3(x)))
        pi = self.flatten(pi)
        if training:
            print(np.shape(pi))
        pi = self.dense_pi(pi)

        if training:
            print(np.shape(pi))

        pi = self.softmax(pi)

        return pi


class ResNN2D():
    def __init__(self, args):
        self.args = args
        self.nn = NN(args)

    def train(self, examples, logfile):
        optimizer = optim.Adam(self.nn.parameters())
        criterion = nn.MSELoss()

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            logfile.write('EPOCH ::: ' + str(epoch + 1) + '\n')
            self.nn.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / self.args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args.batch_size):
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                states, pi = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                states = states.reshape(-1,2,19,5)
                target_pi = torch.FloatTensor(pi)

                states, target_pi = Variable(states), Variable(target_pi)

                # measure data loading time
                data_time.update(time.time() - end)

                optimizer.zero_grad()

                out_pi = self.nn(states)
                # l_pi = criterion(out_pi, target_pi)
                l_pi = self.loss_pi(target_pi, out_pi)
                pi_losses.update(l_pi.item(), states.size(0))

                l_pi.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} \n'.format(
                    batch=batch_idx,
                    size=int(len(examples) / self.args.batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                )
                logfile.write("Training Net ")
                bar.next()
                logfile.write(bar.suffix)
            bar.finish()

    def predict(self, inputToModel):
        inputToModel = torch.FloatTensor(inputToModel.astype(np.float64))
        with torch.no_grad():
            inputToModel = Variable(inputToModel)

        self.nn.eval()
        preds = self.nn(inputToModel)
        return preds

    def loss_pi(self, targets, outputs):
        # outputs = outputs.view(-1, 21, 18)
        # targets = targets.view(-1, 21, 18)
        # return -torch.sum(targets.data.cuda() * outputs) / targets.size()[0]
        return -torch.sum((targets.data - outputs) ** 2) / targets.size()[0]

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