import math
import os
from urllib.request import Request, urlopen

from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtGui import QPixmap, QFont, QPainterPath, QPainter
from PyQt6.QtWidgets import *
from hearthstone.enums import Zone as HS_Zone
from PyQt6.QtCore import Qt

from fireplace.card import Spell
from ui.animations import MoveCardAnim, DeathCardAnim

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

FONT = "Belwe Bd BT Alt Style [Rus by m"


class Zone:
    def __init__(self, x, y, count=0):
        self.x = x
        self.y = y
        self.count = count
        self.cards = []


class OutlinedLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        off = 10
        painter = QPainter(self)
        path = QPainterPath()
        draw_font = self.font()
        path.addText(off, draw_font.pointSize() + off, draw_font, self.text())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.strokePath(path, QPen(QColor(Qt.GlobalColor.black), 2))
        painter.fillPath(path, QBrush(Qt.GlobalColor.white))
        size = path.boundingRect().size().toSize()
        self.resize(size.width() + off * 2, size.height() + off * 2)


class QCard(QLabel):
    # TODO:
    FONT_SIZE = 21

    def __init__(self, qwindow, width, height, entity):
        super().__init__(qwindow)

        self.width = width
        self.height = height

        # self.label = QLabel(self)
        self.card_overlay = QLabel(self)
        # self.at = OutlinedLabel(self)
        # self.hp = OutlinedLabel(self)
        self.cost = OutlinedLabel(self)

        self.cost.setText(str(entity.cost))
        self.cost.move(int(self.width * 0.031), int(self.height * 0.067))
        self.cost.setFont(QFont(FONT, self.FONT_SIZE - 1))
        self.cost.setStyleSheet("color:white;")
        self.cost.setAlignment(Qt.AlignmentFlag.AlignCenter)

        overlay_path = "ui/images/spell.png"

        if type(entity) != Spell:
            self.at = OutlinedLabel(self)
            self.hp = OutlinedLabel(self)
            self.hp.setText(str(entity.health))
            self.at.setText(str(entity._atk))

            self.at.setFont(QFont(FONT, self.FONT_SIZE - 2))
            self.at.setGeometry(QtCore.QRect(int(self.width * 0.04), int(self.height * 0.715), 500, 500))
            self.at.setStyleSheet("color:white;")

            self.hp.move(int(self.width * 0.70), int(self.height * 0.715))
            self.hp.setFont(QFont(FONT, self.FONT_SIZE - 2))
            self.hp.setStyleSheet("color:white;")
            self.hp.setAlignment(Qt.AlignmentFlag.AlignCenter)

            overlay_path = "ui/images/minion.png"

        self.card_overlay.setPixmap(QPixmap(overlay_path))
        self.card_overlay.move(int(self.width * 0.045), int(self.height * 0.07))
        self.card_overlay.setScaledContents(True)
        self.card_overlay.resize(self.width * 0.9, self.height * 0.85)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Game Ep")

        self.void_size = 20
        self.card_width = 128
        self.card_height = 194

        self.entities = {'Deck1': Zone(1000, 600, 0), 'Deck2': Zone(1000, 50, 0),
                         'Hand1': Zone(200, 600, 0),
                         'Hand2': Zone(200, 50, 0),
                         'Field1': Zone(200, 430), 'Field2': Zone(200, 240)}
            # ,
            #              'Face1': Zone(0, 0), 'Face2': Zone(0, 600)}

        self.id_list = []
        self.anims = {}
        self.start_timer()
        self.resize(1300, 800)
        self.show()

    def add_animation(self, entity, animation):
        self.anims.setdefault(entity.objectName(), []).append(animation)

    def add_entity_to_hand(self, entity):
        player_name = entity.controller.name
        hand = "Hand" + player_name[-1]
        self.entities[hand].count += 1
        card_position = len(self.entities[hand].cards) - 1
        cardID = entity.id
        entity_zone_pos = entity.zone_position - 1

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
        self.entities[hand].cards.insert(entity_zone_pos, QCard(self, self.card_width, self.card_height, entity))
        # self.entity[hand].cards.append(QLabel(self))
        self.entities[hand].cards[entity_zone_pos].setObjectName(str(entity.uuid))
        self.entities[hand].cards[entity_zone_pos].setScaledContents(True)
        self.entities[hand].cards[entity_zone_pos].setPixmap(QPixmap(os.path.join(cards_path,
                                                                                  cardID + ".png")))
        self.entities[hand].cards[entity_zone_pos].resize(self.card_width, self.card_height)
        self.reorganise(hand)
        self.entities[entity.uuid] = self.entities[hand].cards[entity_zone_pos]
        self.entities[hand].cards[entity_zone_pos].show()

    def change_zone(self, entity, zoneID_from, zoneID_to):  # cardID: CardID, zoneID: ZoneID,
        player_name = entity.controller.name
        zoneID_from = self.get_zone(zoneID_from, player_name)
        zoneID_to = self.get_zone(zoneID_to, player_name)

        card = self.entities[entity.uuid]  # cardPos: original position
        self.entities[zoneID_from].cards.remove(card)
        self.entities[zoneID_from].count -= 1
        self.reorganise(zoneID_from)
        position = entity.zone_position - 1
        if position == -1:
            # default
            cord_x = self.entities[zoneID_to].x + (self.entities[zoneID_to].count * (self.card_width + self.void_size))
            self.add_animation(card, MoveCardAnim(card, cord_x, self.entities[zoneID_to].y))
            self.entities[zoneID_to].cards.append(card)
        else:
            # moving to another position
            cord_x = self.entities[zoneID_to].x + (position * (self.card_width + self.void_size))
            self.add_animation(card, MoveCardAnim(card, cord_x, self.entities[zoneID_to].y))
            self.entities[zoneID_to].cards.insert(position, card)

        self.entities[zoneID_to].count += 1
        self.reorganise(zoneID_to)

        # print("card started moving")

    @staticmethod
    def get_zone(zone, player_name):
        _zone = ""
        if zone == HS_Zone.HAND:
            _zone = "Hand" + player_name[-1]
        elif zone == HS_Zone.PLAY:
            _zone = "Field" + player_name[-1]
        return _zone

    def remove_entity(self, entity, prev_zone):
        player_name = entity.controller.name
        zone = self.get_zone(prev_zone, player_name)
        self.entities[zone].count -= 1
        # self.entity[hand].cards
        label = self.entities[entity.uuid]
        # label.clear()
        # sip.delete(label)
        self.entities[zone].cards.remove(label)
        del self.entities[entity.uuid]

        # ql = QLabel(self)
        # ql.layout().removeWidget(ql)
        self.add_animation(label, DeathCardAnim(label))
        # label.clear()
        self.reorganise(zone)
        # self.entities[hand].cards[entity.zone_position - 1].clear()
        # del self.entities[hand].cards[entity.zone_position - 1]

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
            # if "Hand" in zoneID:
            i -= 1
            if card.x != self.entities[zoneID].x + (i * (self.card_width + self.void_size)):
                self.add_animation(card, MoveCardAnim(card, self.entities[zoneID].x + (
                        i * (self.card_width + self.void_size)),
                                                      self.entities[zoneID].y))

    def render_hand(self, hand):
        for card_position, card in enumerate(self.entities[hand].cards):
            card.move(self.entities[hand].x + 140 * (card_position - 1), self.entities[hand].y)

    def start_timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.animation)
        timer.start(10)  # 50 for debugging

    def animation(self):
        for id in list(self.anims.keys()):
            if self.anims[id][0].steps >= 20:
                self.anims[id][0].last_step()
                if len(self.anims[id]) == 1:
                    del self.anims[id]
                else:
                    self.anims[id].pop(0)
                # print("card moved")
                continue
            self.anims[id][0].step()
        self.update()

    def attack(self, source, target):
        zoneID_from = self.get_zone(HS_Zone.PLAY, source.controller.name)
        zoneID_to = self.get_zone(HS_Zone.PLAY, target.controller.name)

        card1 = self.entities[source.uuid]
        card1.raise_()
        # cord_x_to = self.entities[zoneID_to].x + (cardPos2 * (self.card_width + self.void_size))
        # cord_x_from = self.entities[zoneID_from].x + (cardPos1 * (self.card_width + self.void_size))
        cord_x_to = self.entities[zoneID_to].x + (target.zone_position * (self.card_width + self.void_size))
        cord_x_from = self.entities[zoneID_from].x + (source.zone_position * (self.card_width + self.void_size))
        self.add_animation(card1, MoveCardAnim(card1, cord_x_to, self.entities[zoneID_to].y))
        self.add_animation(card1, MoveCardAnim(card1, cord_x_from, self.entities[zoneID_from].y))

    # def send_to_graveyard(self, entity, prev_zone):
    #     player_name = entity.controller.name
    #     zone = self.get_zone(prev_zone, player_name)
    #     self.change_zone(entity, zone, "Graveyard" + player_name[-1])
    #     label = self.entities[entity.uuid]
    #     label.setPixmap(QtGui.QPixmap(os.path.join(cards_path, "card_back.png")))
