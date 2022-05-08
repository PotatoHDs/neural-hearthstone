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


class QCard(QFrame):
    # TODO:
    FONT_SIZE = 21

    def __init__(self, qwindow, width, height):
        super().__init__(qwindow)

        self.width = width
        self.height = height

        self.label = QLabel(self)
        self.at = OutlinedLabel(self)
        self.hp = OutlinedLabel(self)
        self.cost = OutlinedLabel(self)

        self.cost.setText("0")
        self.hp.setText("1")
        self.at.setText("1")

        # eff = QGraphicsDropShadowEffect(self)
        # eff.setColor(QColor.black(QColor()))
        # eff.setOffset(0, 0)
        # eff.setBlurRadius(2)
        # eff.setBlurRadius(5)

        self.cost.move(int(self.width * 0.03), int(self.height * 0.075))
        self.cost.setFont(QFont(FONT, self.FONT_SIZE))
        self.cost.setStyleSheet("color:white;")
        # self.cost.setGraphicsEffect(eff)

        self.at.move(int(self.width * 0.04), int(self.height * 0.715))
        self.at.setFont(QFont(FONT, self.FONT_SIZE - 2))
        self.cost.setStyleSheet("color:white;")

        self.hp.move(int(self.width * 0.70), int(self.height * 0.715))
        self.hp.setFont(QFont(FONT, self.FONT_SIZE - 2))
        self.cost.setStyleSheet("color:white;")

    def setPixmap(self, pixmap):
        self.label.setPixmap(pixmap)
        pass

    def setScaledContents(self, scaled):
        self.label.setScaledContents(scaled)
        pass

    def resize(self, w: int, h: int) -> None:
        super().resize(w, h)
        self.label.resize(w, h)

    # def resize(self, width, length):
    #     self.label.resize(width, length)


class MainWindow(QMainWindow):
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
        self.entities[hand].cards.insert(entity_zone_pos, QCard(self, self.card_width, self.card_height))
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

        self.add_animation(label, DeathCardAnim(label))
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
            if card.x != self.entities[zoneID].x + ((i - 1) * (self.card_width + self.void_size)):
                self.add_animation(card, MoveCardAnim(card, self.entities[zoneID].x + (
                        (i - 1) * (self.card_width + self.void_size)),
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

    def attack(self, zoneID_from, zoneID_to, source, target):
        zoneID_from = self.get_zone(zoneID_from, source.controller.name)
        zoneID_to = self.get_zone(zoneID_to, target.controller.name)

        card1 = self.entities[source.uuid]
        card1.raise_()
        # cord_x_to = self.entities[zoneID_to].x + (cardPos2 * (self.card_width + self.void_size))
        # cord_x_from = self.entities[zoneID_from].x + (cardPos1 * (self.card_width + self.void_size))
        cord_x_to = self.entities[zoneID_to].x + (1 * (self.card_width + self.void_size))
        cord_x_from = self.entities[zoneID_from].x + (1 * (self.card_width + self.void_size))
        self.add_animation(card1, MoveCardAnim(card1, cord_x_to, self.entities[zoneID_to].y))
        self.add_animation(card1, MoveCardAnim(card1, cord_x_from, self.entities[zoneID_from].y))
