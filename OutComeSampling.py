import numpy as np
from random import shuffle
import random
import time
from numpy.random import choice


class Kunh:

    def __init__(self):
        self.nodeMap = {}
        self.current_player = 0
        self.deck = np.array([0, 1, 2])
        self.n_actions = 2
        self.current_player = 0
#         Exploration parameter
        self.epsilon = 0.14

    def train(self, n_iterations=10000):
        for i in range(n_iterations):
            if i % 1000 == 0:
                print(i)
            # Regrets strategies half way through
            if i == n_iterations // 2:
                for _, v in self.nodeMap.items():
                    v.strategy_sum = np.zeros(v.n_actions)

            shuffle(self.deck)
            for j in range(2):
                self.current_player = j
                self.cfr('', 1, 1, 1)
        display_results(self.nodeMap)

    def cfr(self, history, p1_reach, p2_reach, sample_reach):
        n = len(history)
        player = n % 2
        player_card = self.deck[0] if player == 0 else self.deck[1]
        if self.is_terminal(history):
            card_opponent = self.deck[1] if player == 0 else self.deck[0]
            reward = self.get_reward(history, player_card, card_opponent)
            return reward / sample_reach, 1
        
        node = self.get_node(player_card, history)
        strategy = node.get_strategy()
        if player == self.current_player:
            # epsilon-on-policy exploration
            probability = self.sample_strategy(strategy)
        else:
            probability = node.strategy
            
        act = node.get_action(probability)
        next_history = history + node.action_dict[act]
        if player == 0:
            util, p_tail = self.cfr(next_history, p1_reach * node.strategy[act], p2_reach, sample_reach * probability[act])
        else:
            util, p_tail = self.cfr(next_history, p1_reach,  p2_reach * node.strategy[act], sample_reach * probability[act])
        util *= -1
        my_reach = p1_reach if player == 1 else p2_reach
        opp_reach = p2_reach if player == 0 else p1_reach
        if player == self.current_player:
            W = util * opp_reach
            for a in range(len(strategy)):
                regret = W * (1.0 - strategy[act]) * p_tail if a == act else -W * p_tail * strategy[act]
                node.regret_sum[a] += regret
        else:
            for a in range(len(node.strategy_sum)):
                node.strategy_sum[a] += (my_reach * node.strategy[a]) / sample_reach
        return util, p_tail * node.strategy[act]

    def sample_strategy(self, strategy):
        for i in range(len(strategy)):
            strategy[i] = (self.epsilon * np.repeat(1 / self.n_actions, self.n_actions)[i] +
                           (1 - self.epsilon) * strategy[i])
        return strategy

    @staticmethod
    def base_line_child_value(a, sampled_action, value, sample_prob):
        baseline = 0
        if a == sampled_action:
            return baseline + (value - baseline) / sample_prob
        else:
            return baseline

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
        self.strategy = self.regret_sum
        for i in range(len(self.regret_sum)):
            if self.regret_sum[i] < 0:
                self.strategy[i] = 0
        normalizing_sum = sum(self.strategy)
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


def display_results(i_map):
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
    trainer.train(n_iterations=50000)
    print(abs(time1 - time.time()))
