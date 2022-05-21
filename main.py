import json

import rel
import websocket
from PyQt6.QtGui import QFontDatabase
# from hearthstone.enums import BlockType, Zone, Step, PlayState

from Coach import Coach
from Game import GameImp as Game
from NN import NNetWrapper as nn
# from fireplace.actions import Attack, Summon, Hit, EndTurn, Discover, Choice, MulliganChoice, Play, GenericChoice, \
#     BeginTurn, Death, TargetedAction, Activate
# from fireplace.card import HeroPower, Hero, Character
# from fireplace.exceptions import GameOver
# from fireplace.managers import BaseObserver
# from fireplace.player import Player
from ui.ui import MainWindow
from PyQt6.QtWidgets import *
import sys
import re
import os
import time
import math

import torch
import torch.utils.data
from observers import UiObserver, HsObserver


class Args:
    def __init__(self):
        self.n_in = 34 * 16
        self.n_out = 21 * 18

        self.epochs = 200
        self.batch_size = 128
        self.lr = 1e-4

        self.arenaCompare = 10
        self.numIters = 20
        self.numEps = 10
        self.maxlenOfQueue = 1000
        self.numMCTS = 50
        self.numItersForTrainExamplesHistory = 20

        self.tempThreshold = 15
        self.updateThreshold = 0.6

        self.ngpu = 0
        print("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.checkpoint = './temp/'
        self.load_model = True
        self.load_folder_file = ('./temp/', 'temp.pth.tar')
        self.load_examples = ('./temp/', 'checkpoint.pth.tar')
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
        collection.append(cls)

    print(len(collection))

    for card in collection:
        card_id = card.id
        if not os.path.exists(os.path.join("ui/cards", card_id + ".png")):
            print(card)
            req = Request('https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(card_id),
                          headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            with open("ui/cards/{}.png".format(card_id), "wb") as file:
                file.write(webpage)


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    print("Opened connection")


def main():
    args = Args()
    g = Game()
    nnet = nn(args)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.load_train_examples()
    c.learn()
    # obj = Args()
    #
    #
    #
    # setattr(obj, "property_name", "value")
    # getattr(obj, "property_name")


    # def main():
    #     g = Game()
    #     g.init_game()
    #     player = g.game.players[0]
    #     for i in range(len(player.hand)):
    #         print(player.hand[i])
    #         print(player.hand[i].__class__.__name__)
    #         print(player.hand[i].card_class)
    #         print(player.hand[i].type)
    #         print(player.hand[i].cost)

    # card_downloader()
    #
    # app = QApplication(sys.argv)
    #
    # QFontDatabase.addApplicationFont("ui/fonts/belwebdbtaltstylerusbym_bold.otf")
    # fontId =
    # if fontId < 0:
    #     print('font not loaded')
    # families = QFontDatabase.applicationFontFamilies(fontId)
    # print(f'"{families[0]}"')
    # font = QFont(families[0])

    # with open("style.qss", "r") as f:
    #     _style = f.read()
    #     app.setStyleSheet(_style)
    # websocket.enableTrace(True)
    # ws = websocket.WebSocketApp("wss://api.gemini.com/v1/marketdata/BTCUSD",
    #                             on_open=on_open,
    #                             on_message=on_message,
    #                             on_error=on_error,
    #                             on_close=on_close)
    #
    # ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
    # rel.signal(2, rel.abort)  # Keyboard Interrupt
    # rel.dispatch()
    #
    # window = MainWindow()
    # g.init_game(UiObserver(window),HsObserver())
    # g.game.manager.register(UiObserver(window))
    # g.game.manager.register(HsObserver())
    # g.start_game()
    # g.mulligan_choice()
    # g.do_action([0, 0])
    # g.do_action([19, 0])
    # for i in range(0, 4):
    #     g.do_action([0, 0])
    #     g.do_action([19, 0])
    # for i in range(0, 4):
    #     g.do_action([17, 0])
    #     g.do_action([19, 0])
    # tiny fin (or desk imp) attacks
    # g.do_action([10, 0])
    # g.do_action([10, 0])
    # for i in range(120):
    #     try:
    #         g.do_action([10, 0])
    #     except GameOver:
    #         print("Game is over")
    #         break
    # print(g.game.players[0].hand)
    # print(g.game.players[1].hand)
    # app.processEvents()

    # sys.exit(app.exec())


if __name__ == "__main__":
    main()
