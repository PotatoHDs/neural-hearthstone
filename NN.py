import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from utils import Bar, AverageMeter


class SimpleNN(nn.Module):
    def __init__(self, args):
        super(SimpleNN, self).__init__()
        self.ngpu = args.ngpu
        self.dense1 = nn.Linear(args.n_in, 64, bias=False)
        self.dense2 = nn.Linear(64, 128, bias=False)
        self.dense3 = nn.Linear(128, args.n_out, bias=False)
        self.dense4 = nn.Linear(128, 1, bias=False)

        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.flat(x)
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        pi = self.dense3(x)
        v = self.dense4(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class NNetWrapper:
    def __init__(self, args):
        self.nnet = SimpleNN(args)
        self.nnet.to(args.device)
        self.args = args

        if (args.device.type == 'cuda') and (args.ngpu > 1):
            self.nnet = nn.DataParallel(self.nnet, list(range(args.ngpu)))

    def train(self, examples, logfile):
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.epochs):
            # print('EPOCH ::: ' + str(epoch + 1))
            logfile.write('EPOCH ::: ' + str(epoch + 1) + '\n')
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / self.args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args.batch_size):
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64)).unsqueeze(1)
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                states, target_pis, target_vs = Variable(states), Variable(target_pis), Variable(target_vs)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                # pi_losses.update(l_pi.data[0], states.size(0))
                # v_losses.update(l_v.data[0], states.size(0))
                pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}\n'.format(
                    batch=batch_idx,
                    size=int(len(examples) / self.args.batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                logfile.write("Training Net ")
                logfile.write(bar.suffix)
                # bar.next()
            bar.finish()

    def predict(self, state):
        """
        state: np array with state
        """
        # preparing input
        state = torch.FloatTensor(state.astype(np.float64)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            state = Variable(state)
        # state = state.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        pi, v = self.nnet(state)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        outputs = outputs.view(-1, 21, 18)
        targets = targets.view(-1, 21, 18)
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        # return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
