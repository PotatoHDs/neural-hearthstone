import sys
import re
import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import *

from PIL import Image
import os
from ui.animations import MoveCardAnim

cards_path = os.path.join(os.path.abspath(os.getcwd()), "ui", "cards")

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
    def __init__(self, game):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Game Ep")

        self.void_size = 20
        self.card_width = 128
        self.card_length = 194

        self.entity = {'Deck1': Zone(1000, 600, 0), 'Deck2': Zone(1000, 50, 0),
                       'Hand1': Zone(200, 600, len(game.players[0].hand)), 'Hand2': Zone(200, 50, len(game.players[1].hand)),
                       'Field1': Zone(200, 430), 'Field2': Zone(200, 240)}

        self.id_list = []

        for i in range(len(game.players[0].hand)):
            cardID = game.players[0].hand[i].id
            while cardID in self.id_list:
                cardID += "_1"
            self.entity['Hand1'].cards.append(QWidget(self, objectName=cardID))
            self.entity['Hand1'].cards[i].resize(self.card_width, self.card_length)
            self.entity['Hand1'].cards[i].move(self.entity['Hand1'].x + (self.card_width+self.void_size) * i, self.entity['Hand1'].y)
            if not os.path.join(cards_path, game.players[0].hand[i].id):
                req = Request(
                    'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(game.players[0].hand[i].id),
                    headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                with open("ui/cards/{}.png".format(collection[i].id), "wb") as file:
                    file.write(webpage)

            self.entity['Hand1'].cards[i].setStyleSheet(
                "border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path,
                                                                                     game.players[0].hand[i].id)))
            self.entity['Hand1'].cards[i].show()
            print(cardID)

        for i in range(len(game.players[1].hand)):
            cardID = game.players[1].hand[i].id
            while cardID in self.id_list:
                cardID += "_1"
            self.entity['Hand2'].cards.append(QWidget(self, objectName = cardID))
            self.entity['Hand2'].cards[i].resize(self.card_width, self.card_length)
            self.entity['Hand2'].cards[i].move(self.entity['Hand2'].x + 140 * i, self.entity['Hand2'].y)
            if not os.path.join(cards_path, game.players[1].hand[i].id):
                req = Request(
                    'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(game.players[1].hand[i].id),
                    headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                with open("ui/cards/{}.png".format(collection[i].id), "wb") as file:
                    file.write(webpage)
            self.entity['Hand2'].cards[i].setStyleSheet(
                "border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path,
                                                                                     game.players[1].hand[i].id)))
            self.entity['Hand2'].cards[i].show()
            print(cardID)

        self.anims = {}
        self.start_timer()
        self.resize(1300, 800)
        self.show()

    def change_zone(self, cardPos, zoneID_from, zoneID_to, position=None):  # cardID: CardID, zoneID: ZoneID,
        card = self.entity[zoneID_from].cards[cardPos]                      # cardPos: original position
        self.entity[zoneID_from].cards.pop(cardPos)
        self.entity[zoneID_from].count -= 1
        self.reorganise(zoneID_from)

        if position is None:
            # default
            cord_x = self.entity[zoneID_to].x + (self.entity[zoneID_to].count * (self.card_width + self.void_size))
            self.anims.setdefault(card.objectName(), []).append(MoveCardAnim(card, cord_x, self.entity[zoneID_to].y))
            self.entity[zoneID_to].cards.append(card)
        else:
            # moving to another position
            cord_x = self.entity[zoneID_to].x + (position * (self.card_width + self.void_size))
            self.anims.setdefault(card.objectName(), []).append(MoveCardAnim(card, cord_x, self.entity[zoneID_to].y))
            self.entity[zoneID_to].cards.insert(position, card)

        self.entity[zoneID_to].count += 1
        self.reorganise(zoneID_to)

        # print("card started moving")

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
        card.move(self.entity[zone_from].x, self.entity[zone_from].y)
        card.show()
        # print("card appeared")
        self.entity[zone_from].cards.append(card)
        self.entity[zone_from].count += 1
        self.change_zone(0, zone_from, zone_to)

    def reorganise(self, zoneID):
        for i in range(self.entity[zoneID].count):
            card = self.entity[zoneID].cards[i]
            if card.x != self.entity[zoneID].x + (i * (self.card_width + self.void_size)):
                self.anims.setdefault(card.objectName(), []).append(MoveCardAnim(card, self.entity[zoneID].x +
                                                    (i * (self.card_width + self.void_size)), self.entity[zoneID].y))

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
