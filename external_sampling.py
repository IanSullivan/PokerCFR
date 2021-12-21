import numpy as np
from random import shuffle
import random
import time
from numpy.random import choice


class Kunh:

    def __init__(self):
        self.nodeMap = {}
        self.expected_game_value = 0
        self.n_cards = 3
        self.nash_equilibrium = dict()
        self.current_player = 0
        self.deck = np.array([0, 1, 2])
        self.n_actions = 2
        self.current_player = 0
        self.iters = 0
        self.AVERAGE_TYPE = "simple"

    def train(self, n_iterations=10000):
        expected_game_value = 0
        for _ in range(n_iterations):
            self.iters += 1
            # Regrets strategies half way through
            if self.iters == n_iterations // 2:
                for _, v in self.nodeMap.items():
                    v.strategy_sum = np.zeros(v.n_actions)
                    expected_game_value = 0

            for j in range(2):
                self.current_player = j
                shuffle(self.deck)
                expected_game_value += self.cfr('')
                if self.AVERAGE_TYPE == "full":
                    self.update_average('', [1, 1])
        expected_game_value /= n_iterations
        display_results(expected_game_value, self.nodeMap)

    def cfr(self, history):
        n = len(history)
        player = n % 2
        player_card = self.deck[0] if player == 0 else self.deck[1]

        if self.is_terminal(history):
            card_opponent = self.deck[1] if player == 0 else self.deck[0]
            reward = self.get_reward(history, player_card, card_opponent)
            return reward

        node = self.get_node(player_card, history)
        strategy = node.get_strategy()
        action_utils = np.zeros(self.n_actions)
        if player == self.current_player:
            # Counterfactual utility per action.
            for act in range(self.n_actions):
                next_history = history + node.action_dict[act]
                action_utils[act] = -1 * self.cfr(next_history)
            util = np.sum(action_utils * strategy)
            regrets = action_utils - util
            node.regret_sum += regrets
            return util
        else:
            a = node.get_action(strategy)
            next_history = history + node.action_dict[a]
            util = -1 * self.cfr(next_history)
            if self.AVERAGE_TYPE == "simple":
                node.strategy_sum += strategy
            return util

    def update_average(self, history, reach_probs):
        n = len(history)
        player = n % 2
        player_card = self.deck[0] if player == 0 else self.deck[1]
        if self.is_terminal(history):
            return
        # If all the probs are zero, zero strategy accumulated, so just return
        if sum(reach_probs) == 0.0:
            return
        node = self.get_node(player_card, history)
        strategy = node.strategy
        if player == self.current_player:
            new_reach_probs = reach_probs.copy()
            for act in range(self.n_actions):
                new_reach_probs[player] *= strategy[act]
                next_history = history + node.action_dict[act]
                self.update_average(next_history, new_reach_probs)
            node.strategy_sum += reach_probs[player] * strategy
        else:
            a = node.get_action(strategy)
            next_history = history + node.action_dict[a]
            reach_probs[player] *= strategy[a]
            self.update_average(next_history, reach_probs)

    @staticmethod
    def is_terminal(history):
        if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
            return True

    @staticmethod
    def get_reward(history, player_card, opponent_card):
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == "bb"
        if terminal_pass:
            if history[-2:] == 'pp':
                return 1 if player_card > opponent_card else -1
            else:
                return 1
        elif double_bet:
            return 2 if player_card > opponent_card else -2

    def get_node(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
            action_dict = {0: 'p', 1: 'b'}
            info_set = Node(key, action_dict)
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]


class Node:
    def __init__(self, key, action_dict, n_actions=2):
        self.key = key
        self.n_actions = n_actions
        self.action_dict = action_dict
        self.possible_actions = np.arange(self.n_actions)

        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)

        self.strategy = np.repeat(1 / self.n_actions, self.n_actions)
        self.average_strategy = np.repeat(1 / self.n_actions, self.n_actions)

    def get_strategy(self):
        #  regrets are set to zero in cfr+
        self.regret_sum[self.regret_sum < 0] = 0
        normalizing_sum = sum(self.regret_sum)
        self.strategy = self.regret_sum
        if normalizing_sum > 0:
            self.strategy = self.strategy / normalizing_sum
        else:
            self.strategy = np.repeat(1 / self.n_actions, self.n_actions)
        return self.strategy

    def get_action(self, strategy):
        return choice(self.possible_actions, p=strategy)

    def get_average_strategy(self):
        strategy = self.strategy_sum

        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy = strategy / normalizing_sum
        else:
            strategy = np.repeat(1 / self.n_actions, self.n_actions)
        return strategy

    def __str__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)


def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0]) % 2 == 0, sorted_items):
        print(v)
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1, sorted_items):
        print(v)


if __name__ == "__main__":
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=10000)
    print(abs(time1 - time.time()))
