from Coach import Coach
from Game import GameImp as Game
from NN import NNetWrapper as nn
from ui.ui import MainWindow
from PySide6.QtWidgets import *
import sys
import re
import os
import time
import math

import torch
import torch.utils.data


class Args:
    def __init__(self):
        self.n_in = 34 * 16
        self.n_out = 21 * 18

        self.epochs = 50
        self.batch_size = 128
        self.lr = 1e-4

        self.arenaCompare = 40
        self.numIters = 10  # 25
        self.numEps = 25
        self.maxlenOfQueue = 100000
        self.numMCTS = 50  # 1000
        self.numItersForTrainExamplesHistory = 20

        self.tempThreshold = 15
        self.updateThreshold = 0.6

        self.ngpu = 0
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        self.checkpoint = './temp/'
        self.load_model = False
        self.load_folder_file = ('. / temp / ', 'temp.pth.tar')
        self.fireplace_log_enabled = True


def card_downloader():
    from urllib.request import Request, urlopen
    from fireplace import cards
    from hearthstone.enums import CardType
    cards.db.initialize()

    collection = []

    for card in cards.db.keys():
        cls = cards.db[card]
        if not cls.collectible:
            continue
        if cls.type == CardType.HERO:
            # Heroes are collectible...
            continue
        collection.append(cls)

    print(len(collection))

    for i in range(len(collection)):
        print(collection[i])
        req = Request('https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(collection[i].id),
                      headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        with open("ui/cards/{}.png".format(collection[i].id), "wb") as file:
            file.write(webpage)


if __name__ == "__main__":
    # card_downloader()

    args = Args()
    g = Game()
    g.init_game()
    # for i in range(len(g.game.players[0].hand)):
    #     print(g.game.players[0].hand[i])
    # for i in range(len(g.game.players[1].hand)):
    #     print(g.game.players[1].hand[i])

    app = QApplication(sys.argv)

    # with open("style.qss", "r") as f:
    #     _style = f.read()
    #     app.setStyleSheet(_style)

    window = MainWindow(g.game)

    # test actions
    window.summon("VAN_CS2_120", 1)
    window.change_zone(1, "Hand2", "Field2")
    window.change_zone(1, "Hand2", "Field2", 0)
    window.change_zone(0, "Field2", "Hand2")

    sys.exit(app.exec())

    # nnet training
    # nnet = nn(args)

    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    # c = Coach(g, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.load_train_examples()
    # c.learn()
