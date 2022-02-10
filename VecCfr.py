"""
Implementation of vector-based cfr
A public tree is used where only information known to an observer is shown
so things like pot size, who bet, number of players remaining ect
reach probability for each private information is stored in a vector for computing rewards and the strategy sum

two types of reward function, the fast one goes in O(n) time and only requires one loop of the traverser private cards
the second is the slower simpler one, which works in O(n^2)

Time saved doesn't really show up in kuhn poker because there are only 3 cards, but in texas holdem where you have
1369 possible pocket cards going from O(n^2) to O(n) greatly speeds up the time for each iteration
"""

import numpy as np
import time


class VectorKuhn:

    def __init__(self):
        self.nodeMap = {}
        self.deck = np.array([0, 1, 2])
        self.action_dict = {0: 'p', 1: 'b'}
        self.n_actions = 2
        self.n_cards = 3
        self.traverser = 0
        self.expected_value = 0

    def train(self, n_iterations=1000):
        for i in range(n_iterations):
            self.traverser = 0 if i % 2 == 0 else 1
            self.walk_tree('', np.array([1, 1, 1]), np.array([1, 1, 1]))
        display_results(self.nodeMap)

    def walk_tree(self, history, pr_1, pr_2):
        n = len(history)
        player = n % 2
        if self.is_terminal(history):
            return self.get_reward_fast(history, pr_2)
            # return self.get_reward(history, pr_2)

        info_sets = [self.get_node(i, history) for i in range(self.n_cards)]
        utils = [0 for _ in range(self.n_cards)]
        strategies = np.array([node.get_strategy() for node in info_sets])
        action_utils = [[0 for _ in range(self.n_cards)] for _ in range(self.n_actions)]
        for action in self.action_dict:
            next_history = history + self.action_dict[action]
            strategies_taken = np.array([strategy[action] for strategy in strategies])
            if self.traverser == player:
                action_utils[action] = self.walk_tree(next_history, pr_1 * strategies_taken, pr_2)
                utils += action_utils[action] * strategies_taken
            else:
                action_utils[action] = self.walk_tree(next_history, pr_1, pr_2 * strategies_taken)
                utils += action_utils[action]

        if self.traverser == player:
            for i in range(len(info_sets)):
                for action in self.action_dict:
                    info_sets[i].regret_sum[action] += action_utils[action][i] - utils[i]
                    info_sets[i].strategy_sum[action] += strategies[i][action] * pr_1[i]
        return utils

    @staticmethod
    def is_terminal(history):
        if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
            return True

    # Naive way of calculating rewards O(n^2) time
    def get_reward(self, history, opp_reach):
        rewards = [0, 0, 0]
        for player_card in [0, 1, 2]:
            for opponent_card in [0, 1, 2]:
                if opponent_card == player_card:
                    continue
                terminal_pass = history[-1] == 'p'
                double_bet = history[-2:] == "bb"
                if terminal_pass:
                    if history[-2:] == 'pp':
                        reward = 1 if player_card > opponent_card else -1
                        rewards[player_card] += reward * opp_reach[opponent_card]
                    else:
                        player_folded = self.who_folded(history)
                        if self.traverser == player_folded:
                            rewards[player_card] += -1 * opp_reach[opponent_card]
                        else:
                            rewards[player_card] += 1 * opp_reach[opponent_card]
                elif double_bet:
                    reward = 2 if player_card > opponent_card else -2
                    rewards[player_card] += reward * opp_reach[opponent_card]
        return np.array(rewards)

    # Faster way of calculating reward O(n)
    def get_reward_fast(self, history, opp_reach):
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == "bb"
        if terminal_pass:
            if history[-2:] == 'pp':
                return self.show_down(opp_reach, 1)
            else:
                fold_player = self.who_folded(history)
                return self.fold_reward(opp_reach, fold_player)
        elif double_bet:
            return self.show_down(opp_reach, 2)

    @staticmethod
    def who_folded(history):
        # pbp
        if len(history) == 3:
            return 0
        # bp
        elif len(history) == 2:
            return 1

    # End of the game and both players reveal their cards, highest card wins
    @staticmethod
    def show_down(opp_reach, payoff):
        total_prob = sum(opp_reach)
        p_lose = total_prob
        p_win = 0
        reward = [0, 0, 0]
        for card in [0, 1, 2]:
            p_lose -= opp_reach[card]
            reward[card] = (p_win - p_lose) * payoff
            p_win += opp_reach[card]
        return np.array(reward)

    def fold_reward(self, opp_reach, fold_player):
        # payoff is always 1 for a fold in this game
        payoff = -1 if fold_player == self.traverser else 1
        loss_prob = sum(opp_reach)
        rewards = [0, 0, 0]
        for card in [0, 1, 2]:
            loss_prob -= opp_reach[card]
            rewards[card] = payoff * loss_prob
            loss_prob += opp_reach[card]
        return np.array(rewards)

    def get_node(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
            info_set = Node(key)
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]


class Node:
    def __init__(self, key, n_actions=2):
        self.key = key
        self.n_actions = n_actions
        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)
        self.strategy = np.repeat(1/self.n_actions, self.n_actions)

    def get_strategy(self):
        regrets = self.regret_sum.copy()
        regrets[regrets < 0] = 0
        normalizing_sum = sum(regrets)
        if normalizing_sum > 0:
            return regrets / normalizing_sum
        else:
            return np.repeat(1/self.n_actions, self.n_actions)

    def get_average_strategy(self):
        strategy = self.strategy_sum.copy()
        # Re-normalize
        total = sum(strategy)
        strategy /= total
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
    trainer = VectorKuhn()
    trainer.train(n_iterations=1000)
    print(abs(time1 - time.time()))
