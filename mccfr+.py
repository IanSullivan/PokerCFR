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

    def train(self, n_iterations=10000):
        # i_map = {}  # map of information sets
        expected_game_value = 0
        for _ in range(n_iterations):
            self.iters += 1
            # Regrets reset after half way through
            if self.iters == n_iterations//2:
                for _, v in self.nodeMap.items():
                    v.strategy_sum = np.zeros(v.n_actions)
                    expected_game_value = 0
            for j in range(2):
                self.current_player = j
                shuffle(self.deck)
                expected_game_value += self.cfr('', 1, 1, 1)
                # for _, v in self.nodeMap.items():
                #     v.update_strategy()
        print(self.iters)
        expected_game_value /= n_iterations
        display_results(expected_game_value, self.nodeMap)

    def cfr(self, history, pr_1, pr_2, sample_prob):
        n = len(history)
        player = n % 2
        player_card = self.deck[0] if player == 0 else self.deck[1]

        if self.is_terminal(history):
            card_player = self.deck[0] if player == 0 else self.deck[1]
            card_opponent = self.deck[1] if player == 0 else self.deck[0]
            reward = self.get_reward(history, card_player, card_opponent)
            return reward

        node = self.get_node(player_card, history)
        strategy = node.strategy

        # Counterfactual utility per action.
        action_utils = np.zeros(self.n_actions)
        if player == self.current_player:
        # if player != 222:
            if player == 0:
                node.reach_pr += pr_1
            else:
                node.reach_pr += pr_2
            for act in range(self.n_actions):

                p = node.get_p(act)
                random.random()
                if random.random() > p:
                    print(node.strategy_sum[act], p)
                    action_utils[act] = 0
                else:
                    next_history = history + node.action_dict[act]
                    # print(strategy[act])
                    if player == 0:
                        action_utils[act] = -1 * self.cfr(next_history, pr_1 * strategy[act], pr_2, sample_prob * p)
                    else:
                        action_utils[act] = -1 * self.cfr(next_history, pr_1, pr_2 * strategy[act], sample_prob * p)

            # Utility of information set.
            util = np.sum(action_utils * strategy)
            regrets = action_utils - util
            regrets = (pr_2 if player == 0 else pr_1) * regrets
            node.regret_sum += regrets
            node.update_strategy()
        else:
            #  second player, no regrets are calculated only one branch is explore, Monte Carlo
            # at random probability take the greedy path other wise explore based on the strategy
            a = node.get_action(strategy)
            next_history = history + node.action_dict[a]
            util = -1 * self.cfr(next_history, pr_1, pr_2, sample_prob)
        return util

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

        self.strategy = np.repeat(1/self.n_actions, self.n_actions)
        self.average_strategy = np.repeat(1/self.n_actions, self.n_actions)

        self.reach_pr = 0
        self.reach_pr_sum = 0

    def update_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.reach_pr_sum += self.reach_pr
        self.strategy = self.get_strategy()
        self.reach_pr = 0

    def get_strategy(self):
        self.regret_sum[self.regret_sum < 0] = 0
        normalizing_sum = sum(self.regret_sum)
        strategy = self.regret_sum
        if normalizing_sum > 0:
            strategy = strategy / normalizing_sum
        else:
            strategy = np.repeat(1/self.n_actions, self.n_actions)
        return strategy

    def get_p(self, act):

        if self.reach_pr_sum != 0:
            strategy = self.strategy_sum / self.reach_pr_sum
        else:
            strategy = self.strategy_sum

        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy = (1000 + strategy) / (1000 + normalizing_sum)
        else:
            strategy = np.repeat(1 / self.n_actions, self.n_actions)
        return max(strategy[act], 0.05)

    def get_action(self, strategy):
        return choice(self.possible_actions, p=strategy)

    def get_average_strategy(self):
        if self.reach_pr_sum != 0:
            strategy = self.strategy_sum / self.reach_pr_sum
        else:
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
    trainer.train(n_iterations=25000)
    print(abs(time1 - time.time()))
