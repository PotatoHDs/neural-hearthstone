import sys
import re
import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import *
from hearthstone.enums import Zone as HS_Zone

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
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Game Ep")

        self.void_size = 20
        self.card_width = 128
        self.card_length = 194

        self.entities = {'Deck1': Zone(1000, 600, 0), 'Deck2': Zone(1000, 50, 0),
                       'Hand1': Zone(200, 600, 0), 'Hand2': Zone(200, 50, 0),
                       'Field1': Zone(200, 430), 'Field2': Zone(200, 240),
                        'Graveyard1': Zone(1000, 430, 0), 'Graveyard2': Zone(1000, 240, 0)}

        self.id_list = []
        self.anims = {}
        self.start_timer()
        self.resize(1300, 800)
        self.show()

    def init_decks(self, game):
        self.card_back_1 = QLabel(self)
        self.card_back_1.setScaledContents(True)
        self.card_back_1.setPixmap(QtGui.QPixmap(os.path.join(cards_path, "card_back.png")))
        self.card_back_1.resize(self.card_width, self.card_length)
        self.card_back_1.move(self.entities['Deck1'].x, self.entities['Deck1'].y)
        self.card_back_1.show()
        self.card_back_2 = QLabel(self)
        self.card_back_2.setScaledContents(True)
        self.card_back_2.setPixmap(QtGui.QPixmap(os.path.join(cards_path, "card_back.png")))
        self.card_back_2.resize(self.card_width, self.card_length)
        self.card_back_2.move(self.entities['Deck2'].x, self.entities['Deck2'].y)
        self.card_back_2.show()
        for i in range(2):
                for entity in game.players[i].deck:
                    deck = "Deck" + str(i+1)
                    self.entities[deck].cards.append(QLabel(self))
                    self.entities[deck].count += 1
                    self.entities[deck].cards[self.entities[deck].count-1].setObjectName(str(entity.uuid))

    @staticmethod
    def get_zone(zone, player_name):
        _zone = ""
        if zone == HS_Zone.HAND:
            _zone = "Hand" + player_name[-1]
        elif zone == HS_Zone.PLAY:
            _zone = "Field" + player_name[-1]
        return _zone

    def change_zone(self, entity, zoneID_from, zoneID_to):  # cardID: CardID, zoneID: ZoneID,
        player_name = entity.controller.name
        zoneID_from = self.get_zone(zoneID_from, player_name)
        zoneID_to = self.get_zone(zoneID_to, player_name)

        card = self.entities[entity.uuid]
        self.entity[zoneID_from].cards.remove(card)
        self.entity[zoneID_from].count -= 1
        self.reorganise(zoneID_from)

        position = entity.zone_position - 1
        if position is -1:
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

    def summon(self, entity, zoneID):
        player_name = entity.controller.name
        zoneID = self.get_zone(zoneID, player_name)
        cardID = entity.id
        try:
            pos = self.entities[zoneID].cards.index(str(entity.uuid))
        except ValueError:
            print("No such card in Zone")
            return

        if not os.path.exists(os.path.join(cards_path, cardID + ".png")):
            req = Request(
                'https://art.hearthstonejson.com/v1/render/latest/enUS/256x/{}.png'.format(
                    cardID),
                headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            with open("ui/cards/{}.png".format(cardID), "wb") as file:
                file.write(webpage)

        self.entities[zoneID].cards[pos].setScaledContents(True)
        self.entities[zoneID].cards[pos].setPixmap(QtGui.QPixmap(os.path.join(cards_path, cardID + ".png")))
        self.entities[zoneID].cards[pos].resize(self.card_width, self.card_length)
        self.entities[entity.uuid] = self.entities[zoneID].cards[pos]
        self.entities[zoneID].cards[pos].show()

    def remove_entity(self, entity, prev_zone):
        player_name = entity.controller.name
        zone = self.get_zone(prev_zone, player_name)
        self.entities[zone].count -= 1
        # self.entity[hand].cards
        label = self.entities[entity.uuid]
        label.clear()
        self.entities[zone].cards.remove(label)
        del self.entities[entity.uuid]
        self.reorganise(zone)

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

    def add_entity_to_hand(self, entity, zoneID):
        self.summon(entity, zoneID)
        player_name = entity.controller.name
        hand = "Hand" + player_name[-1]
        self.entities[hand].count += 1
        entity_zone_pos = entity.zone_position - 1
        self.entities[hand].cards.insert(entity_zone_pos, self.entities[entity.uuid])
        self.reorganise(hand)

    def attack(self, zoneID_from, zoneID_to, cardPos1, cardPos2):
        card1 = self.entity[zoneID_from].cards[cardPos1]
        card1.raise_()
        cord_x_to = self.entity[zoneID_to].x + (cardPos2 * (self.card_width + self.void_size))
        cord_x_from = self.entity[zoneID_from].x + (cardPos1 * (self.card_width + self.void_size))
        self.anims.setdefault(card1.objectName(), []).append(MoveCardAnim(card1, cord_x_to, self.entity[zoneID_to].y))
        self.anims.setdefault(card1.objectName(), []).append(MoveCardAnim(card1, cord_x_from, self.entity[zoneID_from].y))

    def send_to_graveyard(self, entity, prev_zone):
        player_name = entity.controller.name
        zone = self.get_zone(prev_zone, player_name)
        self.change_zone(entity, zone, "Graveyard" + player_name[-1])
        label = self.entities[entity.uuid]
        label.setPixmap(QtGui.QPixmap(os.path.join(cards_path, "card_back.png")))
