import sys
import re
from urllib.request import Request, urlopen

import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import *

from PIL import Image
import os
from ui.animations import MoveCardAnim

# cards_path = os.path.join(os.path.abspath(os.getcwd()), "ui", "cards")
cards_path = os.path.join("ui", "cards")
print(f"PATH TO CARDS = {cards_path}")

"""
    ZoneID: Deck1, Deck2, Hand1, Hand2, Field1, Field2, Graveyard1, Graveyard2
    
    (known card zones): INVALID, PLAY, DECK, HAND, GRAVEYARD, REMOVEDFROMGAME, SETASIDE, SECRET
    
    Animations: {cardID: [list of animations]}
"""

"""
    Done:
        card movement
        consequential animations
        summoning (should be changed only for cards in deck)
"""


class Zone:
    def __init__(self, x, y, count=0):
        self.x = x
        self.y = y
        self.count = count
        self.cards = []

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Game Ep")

        self.void_size = 20
        self.card_width = 128
        self.card_length = 194

        self.entities = {'Deck1': Zone(1000, 600, 0), 'Deck2': Zone(1000, 50, 0),
                       'Hand1': Zone(200, 600, 0),
                       'Hand2': Zone(200, 50, 0),
                       'Field1': Zone(200, 430), 'Field2': Zone(200, 240)}

        self.id_list = []

        # for i in range(len(game.players[0].hand)):
        #     cardID = game.players[0].hand[i].id
        #     while cardID in self.id_list:
        #         cardID += "_1"
        #     self.entity['Hand1'].cards.append(QLabel(self, objectName=cardID))
        #     self.entity['Hand1'].cards[i].setScaledContents(True)
        #     self.entity['Hand1'].cards[i].setPixmap(QPixmap(os.path.join(cards_path,
        #                                                                  game.players[0].hand[i].id + ".png")))
        #     self.entity['Hand1'].cards[i].resize(self.card_width, self.card_length)
        #     self.entity['Hand1'].cards[i].move(self.entity['Hand1'].x + (self.card_width + self.void_size) * i,
        #                                        self.entity['Hand1'].y)
        #     # print(os.path.join(cards_path, game.players[0].hand[i].id + ".png"))
        #     if not os.path.join(cards_path, game.players[0].hand[i].id + ".png"):
        #         req = Request(
        #             'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(
        #                 game.players[0].hand[i].id),
        #             headers={'User-Agent': 'Mozilla/5.0'})
        #         webpage = urlopen(req).read()
        #         with open("ui/cards/{}.png".format(collection[i].id), "wb") as file:
        #             file.write(webpage)
        #
        #     # print(self.entity['Hand1'].cards[i])
        #     # self.entity['Hand1'].cards[i].setStyleSheet(
        #     #     "border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path,
        #     #                                                                          game.players[0].hand[
        #     #                                                                              i].id + ".png")))
        #
        #     # self.entity['Hand1'].cards[i].setStyleSheet("background-color: red;")
        #
        #     self.entity['Hand1'].cards[i].show()
        #     print(cardID)

        # for i in range(len(game.players[1].hand)):
        #     cardID = game.players[1].hand[i].id
        #     while cardID in self.id_list:
        #         cardID += "_1"
        #     self.entity['Hand2'].cards.append(QLabel(self, objectName=cardID))
        #     self.entity['Hand2'].cards[i].setScaledContents(True)
        #     self.entity['Hand2'].cards[i].setPixmap(QPixmap(os.path.join(cards_path,
        #                                                                  game.players[1].hand[i].id + ".png")))
        #     self.entity['Hand2'].cards[i].resize(self.card_width, self.card_length)
        #     self.entity['Hand2'].cards[i].move(self.entity['Hand2'].x + 140 * i, self.entity['Hand2'].y)
        #     if not os.path.join(cards_path, game.players[1].hand[i].id + ".png"):
        #         req = Request(
        #             'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(
        #                 game.players[1].hand[i].id),
        #             headers={'User-Agent': 'Mozilla/5.0'})
        #         webpage = urlopen(req).read()
        #         with open("ui/cards/{}.png".format(collection[i].id), "wb") as file:
        #             file.write(webpage)
        #     # print(os.path.join(cards_path,
        #     #                    game.players[1].hand[
        #     #                        i].id + ".png"))
        #     # self.entity['Hand2'].cards[i].setStyleSheet(
        #     #     'border-image: url(https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png) 0 0 0 0 stretch stretch;'.format(
        #     #         os.path.join(cards_path, game.players[1].hand[
        #     #             i].id)))
        #
        #     # self.entity['Hand2'].cards[i].setStyleSheet("background-color: red;")
        #
        #     self.entity['Hand2'].cards[i].show()
        #     print(cardID)

        self.anims = {}
        self.start_timer()
        self.resize(1300, 800)
        self.show()
        
    def add_entity_to_hand(self, entity):
        player_name = entity.controller.name
        hand = "Hand" + player_name[-1]
        self.entities[hand].count += 1
        card_position = len(self.entities[hand].cards) - 1
        cardID = entity.id
        entity_zone_pos = entity.zone_position-1

        if not os.path.exists(os.path.join(cards_path, cardID + ".png")):
            req = Request(
                    'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(
                        cardID),
                    headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            with open("ui/cards/{}.png".format(cardID), "wb") as file:
                file.write(webpage)

        # while cardID in self.id_list:
            # cardID += "_1"
        self.entities[hand].cards.insert(entity_zone_pos, QLabel(self))
        # self.entity[hand].cards.append(QLabel(self))
        self.entities[hand].cards[entity_zone_pos].setScaledContents(True)
        self.entities[hand].cards[entity_zone_pos].setPixmap(QPixmap(os.path.join(cards_path,
                                                                                  cardID + ".png")))
        self.entities[hand].cards[entity_zone_pos].resize(self.card_width, self.card_length)
        self.render_hand(hand)
        self.entities[entity.entity_id] = self.entities[hand].cards[entity_zone_pos]
        self.entities[hand].cards[entity_zone_pos].show()

    def change_zone(self, cardPos, zoneID_from, zoneID_to, position=None):  # cardID: CardID, zoneID: ZoneID,
        card = self.entities[zoneID_from].cards[cardPos]  # cardPos: original position
        self.entities[zoneID_from].cards.pop(cardPos)
        self.entities[zoneID_from].count -= 1
        self.reorganise(zoneID_from)

        if position is None:
            # default
            cord_x = self.entities[zoneID_to].x + (self.entities[zoneID_to].count * (self.card_width + self.void_size))
            self.anims.setdefault(card.objectName(), []).append(MoveCardAnim(card, cord_x, self.entities[zoneID_to].y))
            self.entities[zoneID_to].cards.append(card)
        else:
            # moving to another position
            cord_x = self.entities[zoneID_to].x + (position * (self.card_width + self.void_size))
            self.anims.setdefault(card.objectName(), []).append(MoveCardAnim(card, cord_x, self.entities[zoneID_to].y))
            self.entities[zoneID_to].cards.insert(position, card)

        self.entities[zoneID_to].count += 1
        self.reorganise(zoneID_to)

        # print("card started moving")

    def remove_entity_from_hand(self, entity):
        player_name = entity.controller.name
        hand = "Hand" + player_name[-1]
        self.entities[hand].count -= 1
        # self.entity[hand].cards
        label = self.entities[entity.entity_id]
        label.clear()
        self.entities[hand].cards.remove(label)
        del self.entities[entity.entity_id]
        self.render_hand(hand)
        # self.entities[hand].cards[entity.zone_position - 1].clear()
        # del self.entities[hand].cards[entity.zone_position - 1]

    def render_hand(self, hand):
        for card_position, card in enumerate(self.entities[hand].cards):
            card.move(self.entities[hand].x + 140 * (card_position-1), self.entities[hand].y)


    def summon(self, id, player):
        zone_from = "Deck" + str(player)
        zone_to = "Hand" + str(player)
        cardID = id
        while cardID in self.id_list:
            cardID += "_1"
        card = QWidget(self, objectName=cardID)
        card.resize(128, 194)

        if not os.path.join(cards_path, id):
            req = Request(
                'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(id),
                headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            with open("ui/cards/{}.png".format(collection[i].id), "wb") as file:
                file.write(webpage)

        card.setStyleSheet("border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path, id)))
        card.move(self.entities[zone_from].x, self.entities[zone_from].y)
        card.show()
        # print("card appeared")
        self.entities[zone_from].cards.append(card)
        self.entities[zone_from].count += 1
        self.change_zone(0, zone_from, zone_to)

    def reorganise(self, zoneID):
        for i in range(self.entities[zoneID].count):
            card = self.entities[zoneID].cards[i]
            if card.x != self.entities[zoneID].x + (i * (self.card_width + self.void_size)):
                self.anims.setdefault(card.objectName(), []).append(MoveCardAnim(card, self.entities[zoneID].x +
                                                                                 (i * (
                                                                                         self.card_width + self.void_size)),
                                                                                 self.entities[zoneID].y))

    def start_timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.animation)
        timer.start(20)  # 50 for debugging

    def animation(self):
        for id in list(self.anims.keys()):
            if self.anims[id][0].steps == 20:
                self.anims[id][0].last_step()
                if len(self.anims[id]) == 1:
                    del self.anims[id]
                else:
                    self.anims[id].pop(0)
                # print("card moved")
                continue
            self.anims[id][0].step()
        self.update()
