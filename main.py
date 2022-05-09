import json

import rel
import websocket
from PyQt6.QtGui import QFontDatabase
from hearthstone.enums import BlockType, Zone, Step, PlayState

from Coach import Coach
from Game import GameImp as Game
from NN import NNetWrapper as nn
from fireplace.actions import Attack, Summon, Hit, EndTurn, Discover, Choice, MulliganChoice, Play, GenericChoice, \
    BeginTurn, Death
from fireplace.card import HeroPower, Hero, Character
from fireplace.exceptions import GameOver
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
        # if cls.type == CardType.HERO:
        #     # Heroes are collectible...
        #     continue
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
class UiObserver(BaseObserver):
    def __init__(self, window):
        self.window = window

    def change_deck(self, player):
        if player.game.step == Step.BEGIN_DRAW:
            return
        self.window.change_deck_amount(player.name, len(player.deck))
        print(f"Deck changed, \n {player=}\n {player.deck=}\n {len(player.deck)=}")

    def trigger_action(self, action, source, at, *args):
        # print(f"Trigger, \n {action=}\n {source=}\n {at=}\n {args=}")

        if at != 1:
            return
        if type(action) == Attack:
            self.window.attack(args[0], args[1])
            print(f"Attack,\n {action=}\n")
        elif type(action) == Summon and type(args[1]) == Hero:
            self.window.add_hero(args[1])
            print(f"Summon, \n {action=}\n {args[1].controller=}\n {args[1]=}")
        elif type(action) == Summon and type(args[1]) == HeroPower:
            print(f"Summon, \n {action=}\n {args[1].controller=}\n {args[1]=}")
        elif type(action) == EndTurn:
            # self.window.end_turn(args[0].name)
            print(f"End turn, \n {action=}\n {source=}\n {args=}")
            # print(f"Summon, \n {action=}\n {source=}\n {args=}")
        elif type(action) == MulliganChoice:
            self.window.change_deck_amount(source.name, len(source.deck))
            self.window.change_state("Mulligan")
        elif type(action) == BeginTurn:
            self.window.change_state(args[0].name + " turn")
            print(f"Begin turn, \n {action=}\n {source=}\n {args=}")
        elif type(action) == Death and type(args[0]) == Hero:
            state = "Tie..."
            for player in source.game.players:
                if player.playstate == PlayState.WON:
                    state = f"{player.name} wins!"
            self.window.change_state(state)
        # if type(entity) == Hero and type(action) == Summon:
        #
        #     print(f"Summoned Hero,\n {action=}\n {entity=}\n {source=}\n")
        # elif type(entity) == HeroPower and type(action) == Summon:
        #     print(f"Summoned hero power,\n {action=}\n {entity=}\n {source=}\n")
        # elif type(action) == Attack:
        #     print(f"Attacking,\n {action=}\n {entity=}\n {source[0]=}\n")
        pass

    def action_start(self, action_type, source, index, target):
        # if type == BlockType.
        # if action_type == BlockType.ATTACK:
        #     # self.window.attack(source, target)
        #     print(target.zone_position)
        #     print(source.zone_position)
        #     print(f"Action started,\n {action_type=}\n {source=}\n {index=}\n {target=}")
        # elif action_type == BlockType.TRIGGER:
        #     print(f"Action started,\n {action_type=}\n {source=}\n {index=}\n {target=}")
        # print(action_type)
        pass

    def action_end(self, type, source):
        # print(f"Action ended,\n {type=}\n{source=}")
        pass

    def game_step(self, step, next_step):
        print(f"Game step,\n {step=}\n {next_step=}")
        pass

    def new_entity(self, entity):
        if type(entity) == Hero:
            print(f"New entity,\n {entity=}")
        pass

    def start_game(self):
        # print(f"Game started!")
        pass

    def turn(self, player):
        # print(f"Turn, {player=}")
        pass

    def change_card(self, card, field_name, prev_value, curr_value):
        # print(f"Card changed,\n {card=}\n {field_name=}\n {prev_value=}\n {curr_value=}")

        if isinstance(card, Character) and (field_name == "damage" or field_name == "max_health"):
            self.window.change_card(card)
            print(f"Card changed,\n {card=}\n {field_name=}\n {prev_value=}\n {curr_value=}")

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
            if type(card) == HeroPower:
                return
            self.window.remove_entity(card, prev_zone)
            print(f"Changed zone from hand or card died,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        elif prev_zone != Zone.INVALID and prev_zone != Zone.DECK:
            self.window.change_zone(card, prev_zone, zone)
            print(f"Changed zone,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")


class HsObserver(BaseObserver):
    def trigger_action(self, action, source, at, *args):
        # print(f"Trigger, \n {action=}\n {source=}\n {at=}\n {args=}")
        if at != 1:
            return
        print(f"Source:\n {source=}\n {args=}")
        action_type = type(action)
        packet_type = "unknown"
        if action_type == Attack:
            if source.game.current_player.name != "Player1":
                return
            packet_type = "attack"
            # print(f"Attack,\n {action=}\n")
        elif action_type == Play:
            packet_type = "play"
        elif action_type == EndTurn:
            if args[0].name != "Player1":
                return
            packet_type = "endturn"
            # print(f"End turn, \n {action=}\n {source=}\n {args=}")
        elif action_type == Discover:
            packet_type = "discover"
        elif action_type == MulliganChoice:
            if source.name != "Player1":
                return
            packet_type = "mulligan"
        elif action_type == GenericChoice:
            packet_type = "choice"
            # print(f"Choice, \n {action=}\n {source=}\n {args=}")
        if packet_type != "unknown":
            values = [str(el.uuid) for el in args if el is not None and type(el) != int]

            res = {
                "data": {
                    "action": packet_type,
                    "values": values
                }
            }

            json_str = json.dumps(res, indent=4)
            print(json_str)
        # else:
        # if isinstance(action, Choice):
        # print(f"Choice, \n {action=}\n {source=}\n {args=}")

        # elif type(action) == Mulligan

        # print(f"Summon, \n {action=}\n {source=}\n {args=}")
        # if type(entity) == Hero and type(action) == Summon:
        #
        #     print(f"Summoned Hero,\n {action=}\n {entity=}\n {source=}\n")
        # elif type(entity) == HeroPower and type(action) == Summon:
        #     print(f"Summoned hero power,\n {action=}\n {entity=}\n {source=}\n")
        # elif type(action) == Attack:
        #     print(f"Attacking,\n {action=}\n {entity=}\n {source[0]=}\n")
        pass


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    print("Opened connection")


def main():
    card_downloader()

    args = Args()
    g = Game()

    # for i in range(len(g.game.players[0].hand)):
    #     print(g.game.players[0].hand[i])
    # for i in range(len(g.game.players[1].hand)):
    #     print(g.game.players[1].hand[i])

    app = QApplication(sys.argv)

    QFontDatabase.addApplicationFont("ui/fonts/belwebdbtaltstylerusbym_bold.otf")
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
    window = MainWindow()

    g.init_game()
    g.game.manager.register(UiObserver(window))
    g.game.manager.register(HsObserver())
    g.start_game()
    g.mulligan_choice()
    g.do_action([0, 0])
    # g.do_action([0, 0])
    g.do_action([19, 0])
    # g.do_action([0, 0])
    g.do_action([0, 0])
    g.do_action([19, 0])
    # tiny fin (or desk imp) attacks
    # g.do_action([10, 0])
    # g.do_action([10, 0])
    for i in range(120):
        try:
            g.do_action([10, 0])
        except GameOver:
            print("Game is over")
            break
    # print(g.game.players[0].hand)
    # print(g.game.players[1].hand)
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
