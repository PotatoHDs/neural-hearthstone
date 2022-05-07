import math

from PyQt6 import sip


class Anim:
    def init_step(self):
        pass
    def step(self):
        pass
    def last_step(self):
        pass

class MoveCardAnim(Anim):
    def __init__(self, card, x, y):
        self.x = x
        self.y = y
        self.x_step = 0
        self.y_step = 0
        self.x_last = 0
        self.y_last = 0
        self.card = card
        self.steps = 0
        self.started = False

    def init_step(self):
        self.x_step = math.trunc((self.x - self.card.x()) / 20)
        self.y_step = math.trunc((self.y - self.card.y()) / 20)
        self.x_last = int(math.fabs(self.x - self.card.x()) % 20 * math.copysign(1, self.x - self.card.x()))
        self.y_last = int(math.fabs(self.y - self.card.y()) % 20 * math.copysign(1, self.y - self.card.y()))
        self.started = True

        # print(self.x - self.card.x(), self.y - self.card.y(), self.x_step, self.y_step, (self.x - self.card.x()) % 20,
        #       (self.y - self.card.y()) % 20, self.x_last, self.y_last)

    def step(self):
        if not self.started:
            self.init_step()
        self.card.move(self.card.x() + self.x_step, self.card.y() + self.y_step)
        self.steps += 1

    def last_step(self):
        self.card.move(self.card.x() + self.x_last, self.card.y() + self.y_last)

class DeathCardAnim(Anim):
    def __init__(self, card):
        self.steps = 0
        self.card = card
        self.started = False

    def init_step(self):
        self.steps = 20
        self.started = True

    def step(self):
        pass

    def last_step(self):
        sip.delete(self.card)
        pass
