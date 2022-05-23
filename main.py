import json

import rel
import websocket
from PyQt6.QtGui import QFontDatabase
# from hearthstone.enums import BlockType, Zone, Step, PlayState

from Coach import Coach
from Agent import Agent
from Game import GameImp as Game
from NN import NNetWrapper as nn
# from KerasNet import ResNN2D
from PythonNet import ResNN2D
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
import numpy as np
import numpy.ma as ma

import torch
import torch.utils.data
from observers import UiObserver, HsObserver


class Args:
    def __init__(self):
        self.n_in = 34 * 16
        self.n_out = 21 * 18

        self.shape_in = (2,19,5)

        self.epochs = 10
        self.batch_size = 128
        self.lr = 1e-3

        self.arenaCompare = 10
        self.numIters = 50
        self.numEps = 10
        self.maxlenOfQueue = 100000
        self.numMCTS = 10
        self.numItersForTrainExamplesHistory = 30

        self.tempThreshold = 150
        self.updateThreshold = 0.6

        self.ngpu = 0
        print("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.checkpoint = './seq/'
        self.load_model = True
        self.load_folder_file = ('./seq/', 'ResNN2D_best.pth.tar')
        self.load_examples = ('./seq/', 'checkpoint.pth.tar')
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


# def main():
    # args = Args()
    # g = Game()
    # nnet = nn(args)
    #
    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    #
    # c = Coach(g, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.load_train_examples()
    # c.learn()
    # obj = Args()
    #
    #
    #
    # setattr(obj, "property_name", "value")
    # getattr(obj, "property_name")

def play_game(g,nnet,pnet,args):
    players = [nnet, None, pnet]
    if g.game.player_to_start == "Player1":
        cur_player = -1
    else:
        cur_player = 1

    it = 0
    while not g.game.ended or g.game.turn > 180:
        it += 1
        # for MCTS
        # pi = players[cur_player + 1].get_action_prob(temp=0)
        # pi_reshape = np.reshape(pi, (21, 18))
        # action = np.where(pi_reshape  == np.max(pi_reshape ))

        # for NN
        pi = players[cur_player + 1].predict(g.get_state().reshape(-1, 2, 19, 5))
        pi = pi.detach().numpy().reshape(-1)
        pi_reshape = np.reshape(pi, (21, 18))
        valids = np.invert(g.get_valid_moves().astype(bool))
        masked_pi = ma.filled(ma.masked_array(pi_reshape, mask=valids, fill_value=0))
        action = np.where(masked_pi == np.max(masked_pi))

        next_state, cur_player = g.get_next_state(cur_player, (action[0][0], action[1][0]))
    return g.get_game_ended()


def main():
    g = Game()
    args = Args()
    pnet = ResNN2D(args)
    nnet = ResNN2D(args)
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    # agent = Agent(g,nnet,args)
    # agent.learn()

    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("ui/fonts/belwebdbtaltstylerusbym_bold.otf")
    window = MainWindow()

    current_game = g.init_game(UiObserver(window), HsObserver())
    # g.game.manager.register(UiObserver(window))
    # g.game.manager.register(HsObserver())
    g.mulligan_choice()
    g.game.player_to_start = g.game.current_player

    play_game(g,nnet,pnet,args)

    app.processEvents()
    sys.exit(app.exec())

    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
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

    # window = MainWindow()
    # current_game = g.init_game(UiObserver(window), HsObserver())
    # g.game.manager.register(UiObserver(window))
    # g.game.manager.register(HsObserver())
    # g.mulligan_choice()
    # g.game.player_to_start = g.game.current_player

    # arena = Arena(nnet, nnet, g, args)
    # st = arena.play_game(UiObserver(window), HsObserver())

    # app.processEvents()
    # sys.exit(app.exec())


if __name__ == "__main__":
    main()
