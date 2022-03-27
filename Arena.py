from utils import Bar, AverageMeter
import numpy as np
from types import *
import time


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self):
        players = [self.player2, None, self.player1]
        cur_player = 1
        current_game = self.game.getInitGame()
        it = 0
        while not current_game.ended or current_game.turn > 180:
            it += 1
            if type(players[cur_player + 1]) is MethodType:
                action = players[cur_player + 1]()
                next_state, cur_player = self.game.get_next_state(cur_player, (action))
            else:
                pi = players[cur_player + 1]()
                pi_reshape = np.reshape(pi, (21, 18))
                action = np.where(pi_reshape == np.max(pi_reshape))
                next_state, cur_player = self.game.get_next_state(cur_player, (action[0][0], action[1][0]))
        return self.game.get_game_ended()

    def play_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in range(num):
            game_result = self.play_game()
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            game_result = self.play_game()
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=num,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        bar.finish()

        return one_won, two_won, draws
