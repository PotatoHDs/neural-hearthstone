import math
from copy import copy

from PyQt6 import sip


class Anim:
    def step(self):
        pass

    def last_step(self):
        pass


class MoveCardAnim(Anim):
    def __init__(self, card, x, y, _raise = False):
        self.x = x
        self.y = y
        self.x_step = 0
        self.y_step = 0
        self.card = card
        self.steps = 0
        self.started = False
        self._raise = _raise

    def init_step(self):
        if self._raise:
            self.card.raise_()
        self.x_step = math.trunc((self.x - self.card.x()) / 20)
        self.y_step = math.trunc((self.y - self.card.y()) / 20)
        self.started = True

    def step(self):
        if not self.started:
            self.init_step()
        self.card.move(self.card.x() + self.x_step, self.card.y() + self.y_step)
        self.steps += 1

    def last_step(self):
        self.card.move(self.x, self.y)


class DeathCardAnim(Anim):
    def __init__(self, card):
        self.steps = 0
        self.card = card

    def step(self):
        self.steps = 20

    def last_step(self):
        sip.delete(self.card)

class SuperAnim(Anim):
    def __init__(self):
        self.anims = []
        self.steps = 0

    def step(self):
        for anim in self.anims:
            anim.step()
        self.steps+=1

    def last_step(self):
        for anim in self.anims:
            anim.last_step()

class ChangeAnim(Anim):
    def __init__(self, card):
        self.card = card
        self.entity = copy(card.entity)
        self.steps = 0

    def step(self):
        self.card.rerender(self.entity)
        self.steps+=20

    def last_step(self):
        pass

class ChangeTextAnim(Anim):
    def __init__(self, label, text):
        self.label = label
        self.text = text
        self.steps = 0

    def step(self):
        self.steps += 20

    def last_step(self):
        self.label.setText(self.text)



