from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from utils import Bar, AverageMeter


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.args)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.
        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            train_examples: a list of examples of the form (state,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        current_game = self.game.init_game()
        self.game.mulligan_choice()
        self.game.game.player_to_start = self.game.game.current_player
        self.cur_player = 1 if f'{current_game.current_player}' == 'Player1' else -1
        episode_step = 0

        while True:
            episode_step += 1
            temp = int(episode_step < self.args.tempThreshold)

            pi = self.mcts.get_action_prob(temp=temp)
            print("Simulated {} step".format(episode_step))
            pi_reshape = np.reshape(pi, (21, 18))
            s = self.game.get_state()
            train_examples.append([s, self.cur_player, pi, None])
            action = np.random.choice(len(pi), p=pi)
            a, b = np.unravel_index(action, pi_reshape.shape)
            print(a, b)
            current_game, self.cur_player = self.game.get_next_state(self.cur_player, (a, b))

            r = self.game.get_game_ended()

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.cur_player))) for x in train_examples]

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            print('------ITER ' + str(i) + '------')
            if not self.skipFirstSelfPlay or i > 1:
                iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iteration_train_examples += self.execute_episode()
                    print("Executed {} eps".format(eps))
                    # print(iteration_train_examples)
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}\n'.format(
                        eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                self.trainExamplesHistory.append(iteration_train_examples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest train_examples")
                self.trainExamplesHistory.pop(0)
            self.save_train_examples(i - 1)

            train_examples = []
            for e in self.trainExamplesHistory:
                train_examples.extend(e)
            shuffle(train_examples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(pmcts.get_action_prob(temp=0), nmcts.get_action_prob(temp=0), self.game)
            pwins, nwins, draws = arena.play_games(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

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
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
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
