from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
import numpy.ma as ma
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from utils import Bar, AverageMeter
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import *
from ui.ui import MainWindow
from observers import UiObserver, HsObserver
from datetime import date

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import csv


class Agent:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.target_nnet = self.nnet.__class__(self.args)  # the competitor network
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def execute_episode(self):
        train_examples = []
        current_game = self.game.init_game()
        self.game.mulligan_choice()
        self.game.game.player_to_start = self.game.game.current_player
        self.cur_player = 1 if f'{current_game.current_player}' == 'Player1' else -1
        episode_step = 0

        while True:
            episode_step += 1

            pi = self.nnet.predict(self.game.get_state().reshape(-1,2,19,5))
            softmax_pi = F.softmax(pi,dim=1)
            # if episode_step % 10 == 0:
            #     print("Simulated {} step".format(episode_step))

            pi = pi.detach().numpy().reshape(-1)
            softmax_pi = softmax_pi.detach().numpy().reshape(21,18)
            pi_reshape = np.reshape(pi, (21, 18))
            s = self.game.get_state()
            valids = np.invert(self.game.get_valid_moves().astype(bool))
            masked_pi = ma.filled(ma.masked_array(pi_reshape, mask=valids, fill_value=0)).reshape(-1)
            masked_pi_softmax = ma.filled(ma.masked_array(softmax_pi, mask=valids, fill_value=0)).reshape(1,-1)
            # softmax_pi = F.softmax(torch.tensor(masked_pi).unsqueeze(0),dim=1)
            action = int(torch.tensor(masked_pi_softmax).max(1)[1].detach())
            # print(masked_pi_softmax)
            # masked_pi_reshape = masked_pi.reshape(-1)
            # counts_sum = float(sum(masked_pi_reshape))
            # print(counts_sum)
            # if counts_sum == 0:
            #     new_valids = np.array(self.game.get_valid_moves().reshape(-1))
            #     masked_pi_renorm = [x / float(sum(new_valids)) for x in new_valids]
            #     action = np.random.choice(len(masked_pi_reshape), p=masked_pi_renorm)
            # else:
            #     masked_pi_renorm = [x / counts_sum for x in masked_pi_reshape]
            #     # print(masked_pi_renorm)
            #     action = np.random.choice(len(masked_pi_renorm), p=masked_pi_renorm)
            a, b = np.unravel_index(action, pi_reshape.shape)
            player = self.cur_player
            cur_game, self.cur_player = self.game.get_next_state(self.cur_player, (a, b))

            r = self.game.get_game_ended()
            train_examples.append([s, player, masked_pi, a, b, action, cur_game, r])

            if r != 0:
                return [(x[0], x[6], x[2].reshape(-1), x[5], (-1) ** (x[1] != self.cur_player)) for x in train_examples]

    def learn(self):
        first = False
        self.target_nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

        for i in range(1, self.args.epochs + 1):
            logs = []
            f = open(f'losses_{i}', 'w')
            writer = csv.writer(f)
            print('------Eps ' + str(i) + '------')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            if not self.skipFirstSelfPlay or i > 1:
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.itersInEps)
                end = time.time()

                for eps in range(self.args.itersInEps):
                    self.trainExamplesHistory.append(self.execute_episode())
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Play Time: {et:.3f}s | Total: {total:}\n'.format(
                        eps=eps + 1, maxeps=self.args.itersInEps, et=eps_time.avg,
                        total=bar.elapsed_td)
                    bar.next()

                    if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                        print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                              " => remove the oldest train_examples")
                        self.trainExamplesHistory.pop(0)
                    train_examples = []
                    for e in self.trainExamplesHistory:
                        train_examples.extend(e)
                    shuffle(train_examples)

                    if first:
                        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                        self.target_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

                    self.nnet_train(train_examples, logs)

                bar.finish()

            writer.writerow(logs)
            f.close()

            first = False
            self.target_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.target_nnet, self.nnet, self.game, self.args)
            pwins, nwins, draws = arena.play_games(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def nnet_train(self, examples, logs):
        if len(examples) < self.args.batch_size:
            return
        # optimizer = optim.RMSprop(self.nnet.nn.parameters())
        optimizer = optim.Adam(self.nnet.nn.parameters())

        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        end = time.time()

        bar = Bar('Training Net', max=int(len(examples) / self.args.batch_size))
        batch_idx = 0

        while batch_idx < int(len(examples) / self.args.batch_size):
            sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
            states, next_states, pi, a, reward = list(zip(*[examples[i] for i in sample_ids]))
            states = torch.FloatTensor(np.array(states).astype(np.float64), device=self.args.device)
            states = states.reshape(-1,2,19,5)
            states = Variable(states)
            # next_states = torch.FloatTensor(np.array(next_states).astype(np.float64), device=self.args.device)
            # next_states = np.array(next_states).astype(np.float64)
            reward = torch.FloatTensor(np.array(reward).astype(np.float64))
            action = torch.tensor(a).unsqueeze(0)

            # measure data loading time
            data_time.update(time.time() - end)

            non_term_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=self.args.device, dtype=torch.bool)
            next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None]).astype(np.float64)).reshape(-1, 2, 19, 5)
            # non_term_next_states = torch.cat([s for s in next_states if s is not None])

            state_action_values = self.nnet.nn(states).gather(1, action)

            reward_batch = torch.FloatTensor(reward)/10
            next_state_values = reward
            next_state_values[non_term_mask] = self.target_nnet.nn(next_states).max(1)[0].detach()
            expected_state_action_values = ((next_state_values + reward_batch) * self.args.gamma)

            # print(state_action_values)
            # print(expected_state_action_values)

            criterion = nn.SmoothL1Loss()
            # criterion = nn.HuberLoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))
            pi_losses.update(loss.item(), states.size(0))

            optimizer.zero_grad()
            loss.backward()
            # for param in self.nnet.nn.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            batch_idx += 1

            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | Loss_pi: {lpi:.4f} \n'.format(
                batch=batch_idx,
                size=int(len(examples) / self.args.batch_size),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                lpi=pi_losses.avg,
            )
            bar.next()
            logs.append(pi_losses.avg)
        bar.finish()

    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def load_train_examples(self):
        model_file = os.path.join(self.args.load_examples[0], self.args.load_examples[1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examples_file, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
