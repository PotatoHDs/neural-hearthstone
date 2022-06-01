from .utils import CardList

class Deck(CardList):
	MAX_CARDS = 30
	MAX_UNIQUE_CARDS = 2
	MAX_UNIQUE_LEGENDARIES = 1

	def __init__(self, cards=None, player=None):
		super().__init__(cards or [])
		self.hero = None
		self.player = player

	def __repr__(self):
		return "<Deck (%i cards)>" % (len(self))

	def append(self, __object):
		super().append(__object)
		self.broadcast()

	def remove(self, __object):
		super().remove(__object)
		self.broadcast()

	def pop(self, __index):
		super().pop(__index)
		self.broadcast()

	def insert(self, __index, __object):
		super().insert(__index, __object)
		self.broadcast()

	def broadcast(self):
		if self.player is None:
			return
		# if self.player.game is None:
		# 	return
		try:
			self.player.game.manager.change_deck(self.player)
		except AttributeError:
			return
