import copy
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

from fireplace.card import Spell, Hero
from ui.animations import MoveCardAnim, DeathCardAnim, SuperAnim, ChangeAnim, ChangeTextAnim

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
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cards = []

    @property
    def count(self):
        return len(self.cards)


class OutlinedLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textColor = Qt.GlobalColor.white

    def setColor(self, color):
        self.textColor = color
        # self.render()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        off = 10
        painter = QPainter(self)
        path = QPainterPath()
        draw_font = self.font()
        path.addText(off, draw_font.pointSize() + off, draw_font, self.text())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.strokePath(path, QPen(QColor(Qt.GlobalColor.black), 2))
        painter.fillPath(path, QBrush(self.textColor))
        size = path.boundingRect().size().toSize()
        self.resize(size.width() + off * 2, size.height() + off * 2)


class QCard(QLabel):
    # TODO:
    FONT_SIZE = 21

    def __init__(self, qwindow, width, height, entity):
        super().__init__(qwindow)

        self.entity = entity

        self.width = width
        self.height = height

        self.card_overlay = QLabel(self)
        self.cost = OutlinedLabel(self)

        self.cost.setText(str(entity.cost))
        self.cost.move(int(self.width * 0.031), int(self.height * 0.067))
        self.cost.setFont(QFont(FONT, self.FONT_SIZE - 1))
        self.cost.setAlignment(Qt.AlignmentFlag.AlignCenter)

        overlay_path = "ui/images/spell.png"

        if type(entity) != Spell:
            self.at = OutlinedLabel(self)
            self.hp = OutlinedLabel(self)
            self.hp.setText(str(entity.health))
            self.at.setText(str(entity.atk))

            self.at.setFont(QFont(FONT, self.FONT_SIZE - 2))
            self.at.setGeometry(QtCore.QRect(int(self.width * 0.04), int(self.height * 0.715), 500, 500))

            self.hp.move(int(self.width * 0.70), int(self.height * 0.715))
            self.hp.setFont(QFont(FONT, self.FONT_SIZE - 2))
            self.hp.setAlignment(Qt.AlignmentFlag.AlignCenter)

            overlay_path = "ui/images/minion.png"

        self.card_overlay.setPixmap(QPixmap(overlay_path))
        self.card_overlay.move(int(self.width * 0.045), int(self.height * 0.07))
        self.card_overlay.setScaledContents(True)
        self.card_overlay.resize(int(self.width * 0.9), int(self.height * 0.85))

        self.resize(self.width, self.height)

    def rerender(self, entity=None):
        if entity is None:
            entity = self.entity
        self.cost.setText(str(entity.cost))
        if type(entity) != Spell:
            if entity.damage > 0:
                self.hp.setColor(Qt.GlobalColor.red)

            self.hp.setText(str(entity.health))
            self.at.setText(str(entity.atk))
            # print(f"All set, {entity.cost} {entity.health} {entity.atk}")


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Game Ep")

        self.void_size = 20
        self.card_width = 128
        self.card_height = 194

        self.entities = {'Deck1': Zone(1100, 600), 'Deck2': Zone(1100, 50),
                         'Hand1': Zone(200, 600), 'Hand2': Zone(200, 50),
                         'Face1': Zone(52, 600), 'Face2': Zone(52, 50),
                         'Field1': Zone(200, 430), 'Field2': Zone(200, 240)}

        self.turn_label = QLabel(self)
        self.turn_label.resize(400, 50)
        self.turn_label.setFont(QFont(FONT, 20))
        self.turn_label.setText("Game starting")
        self.turn_label.move(1100, 400)
        self.turn_label.show()

        self.deck1_amount_label = OutlinedLabel(self)
        self.deck1_amount_label.resize(400, 50)
        self.deck1_amount_label.setFont(QFont(FONT, 20))
        self.deck1_amount_label.setText("30")
        self.deck1_amount_label.move(self.entities["Deck1"].x - 25 + self.card_width//2,
                                     self.entities["Deck1"].y - 20 + self.card_height//2)
        self.deck1_amount_label.show()

        self.deck2_amount_label = OutlinedLabel(self)
        self.deck2_amount_label.resize(400, 50)
        self.deck2_amount_label.setFont(QFont(FONT, 20))
        self.deck2_amount_label.setText("30")
        self.deck2_amount_label.move(self.entities["Deck2"].x - 25 + self.card_width//2,
                                     self.entities["Deck2"].y - 20 + self.card_height//2)
        self.deck2_amount_label.show()

        self.card_back1_label = QLabel(self)
        self.card_back1_label.resize(self.card_width+10, self.card_height+10)
        self.card_back1_label.move(self.entities["Deck1"].x-5, self.entities["Deck1"].y-5)
        self.card_back1_label.setScaledContents(True)
        self.card_back1_label.setPixmap(QPixmap("ui/images/cardback.png"))

        self.card_back2_label = QLabel(self)
        self.card_back2_label.resize(self.card_width+10, self.card_height+10)
        self.card_back2_label.move(self.entities["Deck2"].x-5, self.entities["Deck2"].y-5)
        self.card_back2_label.setScaledContents(True)
        self.card_back2_label.setPixmap(QPixmap("ui/images/cardback.png"))

        self.id_list = []
        self.anims = []
        self.start_timer()
        self.resize(1300, 800)
        self.show()

    def add_animation(self, animation):
        self.anims.append(animation)

    def change_state(self, text):
        self.add_animation(ChangeTextAnim(self.turn_label, text))

    def add_entity_to_hand(self, entity):
        player_name = entity.controller.name
        hand = "Hand" + player_name[-1]
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
        self.reorganise(hand)
        self.entities[entity.uuid] = self.entities[hand].cards[entity_zone_pos]
        zone = self.entities[self.get_zone(HS_Zone.DECK, entity.controller.name)]
        self.entities[entity.uuid].move(zone.x, zone.y)
        self.entities[hand].cards[entity_zone_pos].show()

        self.card_back1_label.raise_()
        self.card_back2_label.raise_()

        self.deck1_amount_label.raise_()
        self.deck2_amount_label.raise_()

    def change_card(self, entity):
        self.anims.append(ChangeAnim(self.entities[entity.uuid]))
        # self.entities[entity.uuid].rerender(entity)

    def change_deck_amount(self, p_name, cards_amount):
        if p_name[-1] == '1':
            cur = self.deck1_amount_label
        else:
            cur = self.deck2_amount_label
        self.add_animation(ChangeTextAnim(cur, str(cards_amount)))

    def add_hero(self, hero):
        zone = 'Face' + hero.controller.name[-1]
        entity_zone_pos = hero.zone_position - 1
        cardID = hero.id
        self.entities[zone].cards.append(QCard(self, self.card_width, self.card_height, hero))
        self.entities[zone].cards[entity_zone_pos].setObjectName(str(hero.uuid))
        self.entities[zone].cards[entity_zone_pos].setScaledContents(True)
        self.entities[zone].cards[entity_zone_pos].setPixmap(QPixmap(os.path.join(cards_path,
                                                                                  cardID + ".png")))
        self.entities[zone].cards[entity_zone_pos].show()
        self.entities[hero.uuid] = self.entities[zone].cards[entity_zone_pos]

        self.entities[hero.uuid].move(self.entities[zone].x, self.entities[zone].y)
        # self.reorganise(zone)

    def change_zone(self, entity, zoneID_from, zoneID_to):  # cardID: CardID, zoneID: ZoneID,
        player_name = entity.controller.name
        zoneID_from = self.get_zone(zoneID_from, player_name)
        zoneID_to = self.get_zone(zoneID_to, player_name)

        card = self.entities[entity.uuid]  # cardPos: original position
        self.entities[zoneID_from].cards.remove(card)
        self.reorganise(zoneID_from)
        position = entity.zone_position-1

        # moving to another position
        self.entities[zoneID_to].cards.insert(position, card)
        cord_x = self.entities[zoneID_to].x + (position * (self.card_width + self.get_void_size(zoneID_to)))
        self.add_animation(MoveCardAnim(card, cord_x, self.entities[zoneID_to].y))
        self.reorganise(zoneID_to)

        # print("card started moving")

    @staticmethod
    def get_zone(zone, player_name):
        _zone = ""
        if zone == HS_Zone.HAND:
            _zone = "Hand" + player_name[-1]
        elif zone == HS_Zone.PLAY:
            _zone = "Field" + player_name[-1]
        elif zone == HS_Zone.DECK:
            _zone = "Deck" + player_name[-1]
        elif zone == "Face":
            _zone = "Face" + player_name[-1]
        return _zone

    def remove_entity(self, entity, prev_zone):
        player_name = entity.controller.name
        if type(entity) == Hero:
            zone = self.get_zone("Face", player_name)
        else:
            zone = self.get_zone(prev_zone, player_name)
        # self.entity[hand].cards
        label = self.entities[entity.uuid]
        # label.clear()
        # sip.delete(label)
        self.entities[zone].cards.remove(label)
        del self.entities[entity.uuid]

        # ql = QLabel(self)
        # ql.layout().removeWidget(ql)
        self.add_animation(DeathCardAnim(label))
        # label.clear()
        self.reorganise(zone)
        # self.entities[hand].cards[entity.zone_position - 1].clear()
        # del self.entities[hand].cards[entity.zone_position - 1]

    def reorganise(self, zoneID):
        superAnim = SuperAnim()
        for i in range(self.entities[zoneID].count):
            card = self.entities[zoneID].cards[i]
            # if "Hand" in zoneID:
            # i += 1
            vs = self.get_void_size(zoneID)
            if card.x != self.entities[zoneID].x + (i * (self.card_width + vs)):
                superAnim.anims.append(MoveCardAnim(card, self.entities[zoneID].x + (
                        i * (self.card_width + vs)), self.entities[zoneID].y))
        self.add_animation(superAnim)

    def render_hand(self, hand):
        for card_position, card in enumerate(self.entities[hand].cards):
            card.move(self.entities[hand].x + 140 * (card_position - 1), self.entities[hand].y)

    def start_timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.animation)
        timer.start(10)  # 50 for debugging

    def animation(self):
        # for id in list(self.anims.keys()):
        #     if self.anims[id][0].steps >= 20:
        #         self.anims[id][0].last_step()
        #         if len(self.anims[id]) == 1:
        #             del self.anims[id]
        #         else:
        #             self.anims[id].pop(0)
        #         # print("card moved")
        #         continue
        #     self.anims[id][0].step()
        if 0 >= len(self.anims):
            return
        if self.anims[0].steps >= 20:
            self.anims[0].last_step()
            self.anims.pop(0)
            return
        self.anims[0].step()

        self.update()

    def get_void_size(self, zoneID):
        # return self.void_size / (self.entities[zoneID].count * 2 + 10)
        return 1000//(self.entities[zoneID].count+1) - self.card_width

    def attack(self, source, target):
        zoneID_from = self.get_zone(HS_Zone.PLAY, source.controller.name)
        zoneID_to = self.get_zone(HS_Zone.PLAY, target.controller.name)
        target_pos = target.zone_position-1
        source_pos = source.zone_position - 1
        if type(source) == Hero:
            source_pos+=1
            zoneID_from = self.get_zone("Face", source.controller.name)
        if type(target) == Hero:
            target_pos+=1
            zoneID_to = self.get_zone("Face", target.controller.name)

        card1 = self.entities[source.uuid]
        # cord_x_to = self.entities[zoneID_to].x + (cardPos2 * (self.card_width + self.void_size))
        # cord_x_from = self.entities[zoneID_from].x + (cardPos1 * (self.card_width + self.void_size))
        # prev_card_x = card1.x
        # prev_card_y = card1.y
        cord_x_to = self.entities[zoneID_to].x + (target_pos * (self.card_width + self.get_void_size(zoneID_from)))
        cord_x_from = self.entities[zoneID_from].x + (source_pos * (self.card_width + self.get_void_size(zoneID_from)))
        self.add_animation(MoveCardAnim(card1, cord_x_to, self.entities[zoneID_to].y, True))
        self.add_animation(MoveCardAnim(card1, cord_x_from, self.entities[zoneID_from].y))
        # self.add_animation(MoveCardAnim(card1, prev_card_x, prev_card_y))


    # def send_to_graveyard(self, entity, prev_zone):
    #     player_name = entity.controller.name
    #     zone = self.get_zone(prev_zone, player_name)
    #     self.change_zone(entity, zone, "Graveyard" + player_name[-1])
    #     label = self.entities[entity.uuid]
    #     label.setPixmap(QtGui.QPixmap(os.path.join(cards_path, "card_back.png")))
