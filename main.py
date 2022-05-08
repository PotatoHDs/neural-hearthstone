from PyQt6.QtGui import QFontDatabase
from hearthstone.enums import BlockType, Zone

from Coach import Coach
from Game import GameImp as Game
from NN import NNetWrapper as nn
from fireplace.actions import Attack
from fireplace.card import HeroPower, Hero
from fireplace.managers import BaseObserver
from ui.ui import MainWindow
from PyQt6.QtWidgets import *
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

    for card in collection:
        card_id = card.id
        if not os.path.exists(os.path.join("ui/cards", card_id + ".png")):
            print(card)
            req = Request('https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(card_id),
                          headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            with open("ui/cards/{}.png".format(card_id), "wb") as file:
                file.write(webpage)


# noinspection PyMethodMayBeStatic
class NewObserver(BaseObserver):
    def __init__(self, window):
        self.window = window

    def action_start(self, action_type, source, index, target):
        # if type == BlockType.
        if action_type == BlockType.ATTACK:
            print(f"Action started,\n {action_type=}\n {source=}\n {index=}\n {target=}")
        pass

    def action_end(self, type, source):
        # print(f"Action ended,\n {type=}\n{source=}")
        pass

    def game_step(self, step, next_step):
        # print(f"Game step,\n {step=}\n {next_step=}")
        pass

    def new_entity(self, entity):
        # print(f"New entity,\n {entity=}")
        pass

    def start_game(self):
        # print(f"Game started!")
        pass

    def turn(self, player):
        # print(f"Turn, {player=}")
        pass

    def change_zone(self, card, zone, prev_zone):
        # TODO: Fix entity_id of this classes
        # print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        # if type(card) == HeroPower or type(card) == Hero:
        #     return
        if zone == Zone.HAND and card.zone_position != 0:
            self.window.add_entity_to_hand(card)
            print(f"Changed zone to hand,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        elif prev_zone == Zone.HAND and zone == Zone.DECK or prev_zone == Zone.PLAY and zone == Zone.GRAVEYARD:
            self.window.remove_entity(card, prev_zone)
            print(f"Changed zone from hand or card died,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        elif prev_zone != Zone.INVALID:
            self.window.change_zone(card, prev_zone, zone)
            print(f"Changed zone,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")


def main():
    card_downloader()

    args = Args()
    g = Game()

    # for i in range(len(g.game.players[0].hand)):
    #     print(g.game.players[0].hand[i])
    # for i in range(len(g.game.players[1].hand)):
    #     print(g.game.players[1].hand[i])

    app = QApplication(sys.argv)

    fontId = QFontDatabase.addApplicationFont("ui/fonts/belwebdbtaltstylerusbym_bold.otf")
    if fontId < 0:
        print('font not loaded')
    families = QFontDatabase.applicationFontFamilies(fontId)
    # print(f'"{families[0]}"')
    # font = QFont(families[0])

    # with open("style.qss", "r") as f:
    #     _style = f.read()
    #     app.setStyleSheet(_style)

    window = MainWindow()

    g.init_game()
    g.game.manager.observers.append(NewObserver(window))
    g.start_game()
    g.mulligan_choice()
    g.do_action([0, 0])
    g.do_action([0, 0])
    g.do_action([19, 0])
    g.do_action([0, 0])
    g.do_action([0, 0])
    g.do_action([19, 0])
    # tiny fin (or desk imp) attacks
    g.do_action([10, 1])
    g.do_action([19, 0])
    g.do_action([10, 1])
    print(g.game.players[0].hand)
    print(g.game.players[1].hand)
    # app.processEvents()

    # test actions
    # window.summon("VAN_CS2_120", 1)
    # window.change_zone(1, "Hand2", "Field2")
    # window.change_zone(1, "Hand2", "Field2", 0)
    # window.change_zone(0, "Field2", "Hand2")

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


if __name__ == "__main__":
    main()
