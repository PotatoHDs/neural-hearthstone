from Coach import Coach
from Game import GameImp as Game
from NN import NNetWrapper as nn

import torch
import torch.utils.data


class args():
    def __init__(self):
        self.n_in = 34*16
        self.n_out = 21*18

        self.epochs = 50
        self.batch_size = 128
        self.lr = 1e-4

        self.arenaCompare = 40
        self.numIters = 10  #25
        self.numEps = 25
        self.maxlenOfQueue = 100000
        self.numMCTS = 100  #1000
        self.numItersForTrainExamplesHistory = 20

        self.tempThreshold = 15
        self.updateThreshold = 0.6

        self.ngpu = 0
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        self.checkpoint = './temp/'
        self.load_model = False
        self.load_folder_file = ('. / temp / ', 'temp.pth.tar')


if __name__ == "__main__":

    args = args()
    g = Game()
    nnet = nn(args)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
