import sys
import re
import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import *

from PIL import Image
import os

cards_path = os.path.join(os.path.abspath(os.getcwd()), "ui", "cards")


# class CardWidget(QWidget):
#     def __init__(self, name):
#         super().__init__()
#         self.name = name
#         self.setObjectName(name)
#         self.setStyleSheet(
#             "border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path, name)))
#         self.resize(90, 140)


class MainWindow(QMainWindow):
    def __init__(self, game):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Game Ep")

        self.hand_list = []
        for i in range(len(game.players[0].hand)):
            self.hand_list.append(QWidget(self))
            self.hand_list[i].resize(128, 194)
            self.hand_list[i].move(200 + 140 * i, 100)
            self.hand_list[i].setStyleSheet(
                "border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path,
                                                                                     game.players[0].hand[i].id)))
            self.hand_list[i].show()

        self.en_hand_list = []
        for i in range(len(game.players[1].hand)):
            self.en_hand_list.append(QWidget(self))
            self.en_hand_list[i].resize(128, 194)
            self.en_hand_list[i].move(200 + 140 * i, 500)
            self.en_hand_list[i].setStyleSheet(
                "border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path,
                                                                                     game.players[1].hand[i].id)))
            self.en_hand_list[i].show()

        self.button = QPushButton("Do Action", self)
        self.button.setGeometry(700, 300, 200, 50)
        # self.button.clicked.connect(lambda: self.move_card(self.hand_list[0]))

    def change_zone(self, cardID, zoneID_from, zoneID_to, position=None):  # cardID: CardID, zoneID: ZoneID
        card = self.entity[cardID]
        x = card.x()
        y = card.y()

        # default
        cords_x, cords_y, count = self.entity[zoneID_to]
        cords_x = cords_x + (count * (self.card_width + self.void_size))
        self.anim = QtCore.QPropertyAnimation(card, b"pos")
        self.anim.setEndValue(QtCore.QPoint(x + cords_x, y + cords_y))
        self.anim.setDuration(800)
        self.anim.start()

        self.entity[zoneID_from].count -= 1
        self.entity[zoneID_to].count += 1

    # zoneID: {x:x,y:y, count: count}

    def summon(self, id):
        # cardID = #какая то логика с id
        card = QWidget(self)
        card.resize(128, 194)
        card.setStyleSheet("border-image: url({}) 0 0 0 0 stretch stretch;".format(os.path.join(cards_path, id)))
        card.show()
        self.entity.append({CardID: card})