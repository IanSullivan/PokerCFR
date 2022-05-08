import numpy as np


class Exploitability:

    def __init__(self):
        self.nodeMap = {}
        self.expected_game_value = 0
        self.n_cards = 3
        self.action_dict = {0: 'p', 1: 'b'}
        # Pre-compute average strategies with CFR
        self.average_strategies = {'0 ': [0.8, 0.2], "0 pb": [1.0, 0],
                                   "1 ": [1.0, 0.0], "1 pb": [0.47, 0.53],
                                   "2 ": [0.4, 0.6], "2 pb": [0.0, 1.0],

                                   "0 b": [1.0, 0.0], "0 p": [0.67, 0.33],
                                   "1 b": [0.67, 0.33], "1 p": [1.0, 0.0],
                                   "2 b": [0.0, 1.0], "2 p": [0.0, 1.0]}

        self.best_response_strategies = dict()
        self.deck = np.array([0, 1, 2])
        self.n_actions = 2
        self.traverser = 1

    # Walk tree, calc best response
    def walk_tree(self, history, pr_2):
        n = len(history)
        player = n % 2
        if self.is_terminal(history):
            return self.get_reward(history, pr_2)
        strategies = np.array([self.average_strategies[str(card) + " " + history] for card in range(self.n_cards)])
        utils = [0 for _ in range(self.n_cards)]
        action_utils = np.array([[0 for _ in range(self.n_cards)] for _ in range(self.n_actions)], dtype=np.float32)
        for action in self.action_dict:
            next_history = history + self.action_dict[action]
            strategies_taken = np.array([strategy[action] for strategy in strategies])
            if self.traverser == player:
                action_utils[action] = self.walk_tree(next_history, pr_2)
            else:
                action_utils[action] = self.walk_tree(next_history, pr_2 * strategies_taken)
                utils -= action_utils[action] * strategies_taken

        if player == self.traverser:
            for i, value in enumerate(action_utils.T):
                utils[i] = -max(value)
        return utils

    @staticmethod
    def is_terminal(history):
        if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
            return True

    def get_reward(self, history, opp_reach):
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
        self.strategy = np.repeat(1 / self.n_actions, self.n_actions)

    def get_strategy(self):
        regrets = self.regret_sum.copy()
        regrets[regrets < 0] = 0
        normalizing_sum = sum(regrets)
        if normalizing_sum > 0:
            return regrets / normalizing_sum
        else:
            return np.repeat(1 / self.n_actions, self.n_actions)

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


if __name__ == '__main__':
    a = Exploitability()
    startingReaches = [1, 1, 1]
    a.traverser = 0
    p1_average_ev = sum(a.walk_tree("", startingReaches))
    startingReaches = [1, 1, 1]
    a.traverser = 1
    p2_average = sum(a.walk_tree("", startingReaches))
    print((p1_average_ev + p2_average) / 2)
