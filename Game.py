import numpy as np
import copy
import random

from numba import jit

from fireplace import cards
from fireplace.exceptions import GameOver, InvalidAction
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from fireplace.managers import BaseObserver
from hearthstone.enums import CardClass, CardType, State, BlockType, Zone


# noinspection PyMethodMayBeStatic
class NewObserver(BaseObserver):
    def __init__(self, window):
        self.window = window
    def action_start(self, action_type, source, index, target):
        # if type == BlockType.
        if action_type == BlockType.ATTACK:
            print(f"Action started,\n {action_type=}\n {source=}\n {index=}\n {target=}")
        pass

    def action_end(self, type, source):
        # print(f"Action ended,\n {type=}\n{source=}")
        pass

    def game_step(self, step, next_step):
        # print(f"Game step,\n {step=}\n {next_step=}")
        pass

    def new_entity(self, entity):
        # print(f"New entity,\n {entity=}")
        pass

    def start_game(self):
        # print(f"Game started!")
        pass

    def turn(self, player):
        # print(f"Turn, {player=}")
        pass

    def change_zone(self, card, zone, prev_zone):
        # TODO: Fix entity_id of this classes
        # print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        # if type(card) == HeroPower or type(card) == Hero:
        #     return
        if zone == Zone.HAND and card.zone_position != 0:
            self.window.add_entity_to_hand(card)
            print(f"Changed zone to hand,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        elif prev_zone == Zone.HAND and zone == Zone.DECK or prev_zone == Zone.PLAY and zone == Zone.GRAVEYARD:
            self.window.remove_entity(card, prev_zone)
            print(f"Changed zone from hand or card died,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")
        elif prev_zone != Zone.INVALID:
            self.window.change_zone(card, prev_zone, zone)
            print(f"Changed zone,")
            print(f"\n{card=}\n {card.zone_position=}\n {card.controller=}\n {zone=}\n {prev_zone=} ")


class GameImp:
    def __init__(self, game=None):
        self.game = game

    def init_game(self, UiObserver=None, HsObserver=None):
        cards.db.initialize()
        # print('i\'m doing things')

        # c1 = CardClass(random.randint(2, 10))
        # c2 = CardClass(random.randint(2, 10))
        # deck1 = random_draft(c1)
        # deck2 = random_draft(c2)

        # deck1 = ["CORE_LOEA10_3", "BT_257", "BOT_219", "CORE_GVG_044", "GVG_010",
        #           "LETL_834H3", "VAN_CS2_119", "CORE_EX1_194", "AT_101", "OG_094",
        #           "VAN_CS2_120", "DAL_092", "VAN_CS2_004", "LOOT_258", "DRG_239",
        #           "CORE_LOEA10_3", "BT_257", "BOT_219", "CORE_GVG_044", "GVG_010",
        #           "LETL_834H3", "VAN_CS2_119", "CORE_EX1_194", "AT_101", "OG_094",
        #           "VAN_CS2_120", "DAL_092", "VAN_CS2_004", "LOOT_258", "DRG_239"
        #           ]
        deck2 = ["SCH_145", "CORE_LOEA10_3", "ICC_023", "DRG_239", "CFM_334",
                 "DAL_092", "OG_326", "CS2_172", "TU5_CS2_120", "OG_248",
                 "OG_325", "VAN_CS2_182", "GVG_071", "CFM_665", "AT_101",
                 "SCH_145", "CORE_LOEA10_3", "ICC_023", "DRG_239", "CFM_334",
                 "DAL_092", "OG_326", "CS2_172", "TU5_CS2_120", "OG_248",
                 "OG_325", "VAN_CS2_182", "GVG_071", "CFM_665", "AT_101"
                 ]
        deck1 = ["SCH_145", "CORE_LOEA10_3", "ICC_023", "DRG_239", "CFM_334",
                 "DAL_092", "OG_326", "CS2_172", "TU5_CS2_120", "OG_248",
                 "OG_325", "VAN_CS2_182", "GVG_071", "CFM_665", "AT_101",
                 "SCH_145", "CORE_LOEA10_3", "ICC_023", "DRG_239", "CFM_334",
                 "DAL_092", "OG_326", "CS2_172", "TU5_CS2_120", "OG_248",
                 "OG_325", "VAN_CS2_182", "GVG_071", "CFM_665", "AT_101"
                 ]
        # deck1 = ["DAL_092"]*30
        # deck2 = ["DAL_092"]*30
        # deck1 = ["LOOT_258"]*30
        # deck2 = ["LOOT_258"]*30
        # c1 = CardClass(6)
        c2 = CardClass(3)
        c1 = CardClass(3)

        players = []
        players.append(Player("Player1", deck1, c1.default_hero))
        players.append(Player("Player2", deck2, c2.default_hero))
        self.game = Game(players=players)

        if UiObserver:
            self.game.manager.register(UiObserver)
        if HsObserver:
            self.game.manager.register(HsObserver)

        self.game.start()
        return self.game

    def mulligan_choice(self):
        for player in self.game.players:
            cards_to_mulligan = random.sample(player.choice.cards, random.randint(0, 3))
            player.choice.choose(*cards_to_mulligan)

    def get_next_state(self, player, action):
        # game_copy = copy.deepcopy(game)
        try:
            self.do_action(action)
        except GameOver:
            raise GameOver

        next_state = self.get_state()
        if self.get_game_ended()!=0:
            next_state = None

        if action[0] != 19:
            return next_state, player
        else:
            return next_state, -player

    def get_valid_moves(self):
        actions = np.zeros((21, 18), dtype=int)
        player = self.game.current_player
        if player.choice:
            for index, card in enumerate(player.choice.cards):
                actions[20, index] = 1

        else:
            for index, card in enumerate(player.hand):
                if card.is_playable():
                    if card.requires_target():
                        for target, card in enumerate(card.targets):
                            actions[index, target] = 1
                    else:
                        actions[index] = 1
            for position, minion in enumerate(player.field):
                if minion.can_attack():
                    for target, card in enumerate(minion.attack_targets):
                        actions[position + 10, target] = 1
            if player.hero.power.is_usable():
                if player.hero.power.requires_target():
                    for target, card in enumerate(player.hero.power.targets):
                        actions[17, target] = 1
                else:
                    actions[17] = 1
            if player.hero.can_attack():
                for target, card in enumerate(player.hero.attack_targets):
                    actions[18, target] = 1
            actions[19, 0] = 1
        return actions

    def do_action(self, a):
        player = self.game.current_player
        # self.game.process_deaths()
        if not self.game.ended:
            try:
                if 0 <= a[0] <= 9:
                    if player.hand[a[0]].requires_target():
                        player.hand[a[0]].play(target=player.hand[a[0]].targets[a[1]])
                    else:
                        player.hand[a[0]].play()
                elif 10 <= a[0] <= 16:
                    player.field[a[0] - 10].attack(player.field[a[0] - 10].attack_targets[a[1]])
                elif a[0] == 17:
                    if player.hero.power.requires_target():
                        player.hero.power.use(player.hero.power.play_targets[a[1]])
                    else:
                        player.hero.power.use()
                elif a[0] == 18:
                    player.hero.attack(player.hero.attack_targets[a[1]])
                elif a[0] == 19:
                    player.game.end_turn()
                elif a[0] == 20 and not player.choice:
                    player.game.end_turn()
                elif player.choice:
                    player.choice.choose(player.choice.cards[a[1]])
                else:
                    print("Not an appropriate action")
            except InvalidAction:
                print("InvalidAction")
                # player.game.end_turn()
            except IndexError:
                print("IndexError")
                try:
                    player.game.end_turn()
                except GameOver:
                    pass
            except GameOver:
                pass

    # calculating reward
    def get_game_ended(self):
        p1 = self.game.player_to_start

        if p1.playstate == 4:
            return 1
        elif p1.playstate == 5:
            return -1
        elif p1.playstate == 6:
            return 1e-4
        elif self.game.turn > 180:
            self.game.ended = True
            return 1e-4
        return 0

    # getting state for network in array
    # def get_state(self):
    #     s = np.zeros((34, 16), dtype='float')
    #
    #     p1 = self.game.current_player
    #     p2 = p1.opponent
    #
    #     s[0] = p1.hero.health
    #     s[1] = p2.hero.health
    #     s[2] = p1.max_mana
    #     s[3] = p2.max_mana
    #     s[4] = p1.mana
    #
    #     # hero player1
    #     i = 5
    #     s[i, 0] = p1.hero.card_class
    #     s[i, 1] = p1.hero.power.is_usable() * 1
    #     s[i, 2] = p1.hero.power.cost  # add effect
    #     if p1.weapon is None:
    #         s[i, 2:5] = 0
    #     else:
    #         s[i, 3] = 1
    #         s[i, 4] = p1.weapon.damage
    #         s[i, 5] = p1.weapon.durability  # add effect, attack, deathrattle
    #
    #     # hero player2
    #     i = 6
    #     s[i, 0] = p2.hero.card_class
    #     s[i, 1] = p2.hero.power.is_usable() * 1
    #     s[i, 2] = p2.hero.power.cost  # add effect
    #     if p2.weapon is None:
    #         s[i, 2:5] = 0
    #     else:
    #         s[i, 3] = 1
    #         s[i, 4] = p2.weapon.damage
    #         s[i, 5] = p2.weapon.durability  # add effect, attack, deathrattle
    #
    #     i = 7
    #     hand = len(p1.hand)
    #     s[i] = hand
    #     s[i + 1] = len(p2.hand)
    #
    #     # hand
    #     i = 9
    #     for j in range(0, 10):
    #         if j < hand:
    #             s[i + j, 0] = 1
    #             s[i + j, 1] = p1.hand[j].cost
    #             s[i + j, 2] = p1.hand[j].card_class
    #             s[i + j, 3] = p1.hand[j].type
    #             if p1.hand[j].type == 4:
    #                 s[i + j, 4] = p1.hand[j].race
    #                 s[i + j, 5] = p1.hand[j].atk
    #                 s[i + j, 6] = p1.hand[j].max_health
    #                 s[i + j, 7] = p1.hand[j].divine_shield * 1
    #                 s[i + j, 8] = p1.hand[j].has_deathrattle * 1
    #                 s[i + j, 9] = p1.hand[j].has_battlecry * 1
    #                 s[i + j, 10] = p1.hand[j].taunt * 1
    #
    #                 # field
    #     i = 19
    #     f1 = len(p1.field)
    #     for j in range(0, 7):
    #         if j < f1:
    #             s[i + j, 0] = 1
    #             s[i + j, 1] = p1.field[j].health
    #             s[i + j, 2] = p1.field[j].card_class
    #             s[i + j, 3] = p1.field[j].can_attack() * 1
    #
    #             s[i + j, 4] = p1.field[j].race
    #             s[i + j, 5] = p1.field[j].atk
    #             s[i + j, 6] = p1.field[j].max_health
    #             s[i + j, 7] = p1.field[j].divine_shield * 1
    #             s[i + j, 8] = p1.field[j].has_deathrattle * 1
    #             s[i + j, 9] = p1.field[j].has_battlecry * 1
    #             s[i + j, 10] = p1.field[j].taunt * 1
    #             s[i + j, 11] = p1.field[j].stealthed * 1
    #             s[i + j, 12] = p1.field[j].silenced * 1
    #
    #     i = 26
    #     f2 = len(p2.field)
    #     for j in range(0, 7):
    #         if j < f2:
    #             s[i + j, 0] = 1
    #             s[i + j, 1] = p2.field[j].health
    #             s[i + j, 2] = p2.field[j].card_class
    #             s[i + j, 3] = p2.field[j].can_attack() * 1
    #
    #             s[i + j, 4] = p2.field[j].race
    #             s[i + j, 5] = p2.field[j].atk
    #             s[i + j, 6] = p2.field[j].max_health
    #             s[i + j, 7] = p2.field[j].divine_shield * 1
    #             s[i + j, 8] = p2.field[j].has_deathrattle * 1
    #             s[i + j, 9] = p2.field[j].has_battlecry * 1
    #             s[i + j, 10] = p2.field[j].taunt * 1
    #             s[i + j, 11] = p2.field[j].stealthed * 1
    #             s[i + j, 12] = p2.field[j].silenced * 1
    #
    #     return s

    def get_state(self):
        s = np.zeros((2, 19, 5), dtype='float')

        p1 = self.game.current_player
        p2 = p1.opponent

        # hero player1
        s[0, 0, 0] = p1.hero.power.cost /10
        s[0, 0, 1] = p1.hero.card_class /10
        s[0, 0, 2] = p1.hero.power.is_usable() * 1
        s[0, 0, 3] = p1.hero.health / p1.hero.max_health
        s[0, 0, 4] = p1.mana /10

        # hero player2
        s[1, 0, 0] = p2.hero.power.cost /10
        s[1, 0, 1] = p2.hero.card_class /10
        s[1, 0, 2] = p2.hero.power.is_usable() * 1
        s[1, 0, 3] = p2.hero.health / p2.hero.max_health
        s[1, 0, 4] = p2.mana /10

        h1 = len(p1.hand)
        for j in range(0, 10):
            if j < h1:
                s[0, j + 1, 0] = p1.hand[j].cost /10
                s[0, j + 1, 1] = p1.hand[j].card_class /10
                s[0, j + 1, 2] = p1.hand[j].type /10
                if p1.hand[j].type == 4:
                    s[0, j + 1, 3] = p1.hand[j].max_health /10
                    s[0, j + 1, 4] = p1.hand[j].atk /10

        f1 = len(p1.field)
        for j in range(0, 7):
            if j < f1:
                s[0, j + 11, 0] = p1.field[j].health /p1.field[j].max_health
                s[0, j + 11, 1] = p1.field[j].card_class /10
                s[0, j + 11, 2] = p1.field[j].can_attack() * 1
                s[0, j + 11, 3] = p1.field[j].max_health /10
                s[0, j + 11, 4] = p1.field[j].atk /10

        f2 = len(p2.field)
        for j in range(0, 7):
            if j < f2:
                s[1, j + 11, 0] = p2.field[j].health /p2.field[j].max_health
                s[1, j + 11, 1] = p2.field[j].card_class /10
                s[1, j + 11, 2] = p2.field[j].can_attack() * 1
                s[1, j + 11, 3] = p2.field[j].max_health /10
                s[1, j + 11, 4] = p2.field[j].atk /10

        h2 = len(p2.hand)
        for j in range(0, 10):
            if j < h2:
                s[1, j + 1, 0] = p2.hand[j].cost /10
                s[1, j + 1, 1] = p2.hand[j].card_class /10
                s[1, j + 1, 2] = p2.hand[j].type /10
                if p2.hand[j].type == 4:
                    s[1, j + 1, 3] = p2.hand[j].max_health /10
                    s[1, j + 1, 4] = p2.hand[j].atk /10

        return s
