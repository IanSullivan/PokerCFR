import numpy as np


class FictitiousPlay:

    def __init__(self):
        self.nodeMap = {}
        self.expected_game_value = 0
        self.n_cards = 3
        self.action_dict = {0: 'p', 1: 'b'}
        self.best_responses = {}
        # initialize all strategies randomly
        self.average_strategies = {'0 ': [0.5, 0.5], "0 pb": [0.5, 0.5],
                                   "1 ": [0.5, 0.5], "1 pb": [0.5, 0.5],
                                   "2 ": [0.5, 0.5], "2 pb": [0.5, 0.5],

                                   "0 b": [0.5, 0.5], "0 p": [0.5, 0.5],
                                   "1 b": [0.5, 0.5], "1 p": [0.5, 0.5],
                                   "2 b": [0.5, 0.5], "2 p": [0.5, 0.5]
                                }

        self.deck = np.array([0, 1, 2])
        self.n_actions = 2
        self.traverser = 1

    def update_strategies(self, t):
        # Assuming t is the current iteration number and is passed to the method
        for key, strat in self.best_responses.items():
            # Convert both strategies to numpy arrays for vectorized operations
            current_avg_strategy = np.array(self.average_strategies[key])
            best_response_strategy = np.array(strat)

            # Apply the weighted average update formula
            updated_avg_strategy = ((t - 1) / t) * current_avg_strategy + (1 / t) * best_response_strategy

            # Update the average strategy in the dictionary
            self.average_strategies[key] = updated_avg_strategy

    # Walk tree, calc best response
    def walk_tree(self, history, pr_2):
        n = len(history)
        player = n % 2
        if self.is_terminal(history):
            return self.get_reward(history, pr_2)
        strategies = np.array([self.average_strategies[str(card) + " " + history] for card in range(self.n_cards)])
        utils = np.zeros(self.n_cards)
        action_utils = np.array([[0 for _ in range(self.n_cards)] for _ in range(self.n_actions)], dtype=np.float32)
        for action in self.action_dict:
            next_history = history + self.action_dict[action]
            strategies_taken = np.array([strategy[action] for strategy in strategies])
            if self.traverser == player:
                action_utils[action] = self.walk_tree(next_history, pr_2)
            else:
                action_utils[action] = self.walk_tree(next_history, pr_2 * strategies_taken)
                utils += action_utils[action]
        if player == self.traverser:
            #  pass up the best response
            for i, value in enumerate(action_utils.T):
                utils[i] = max(value)
                best_reponse = np.argmax(value)
                self.best_responses[str(i) + " " + history] = [1 if j == best_reponse else 0 for j in range(self.n_actions)]
                if value[0] == value[1]:
                    self.best_responses[str(i) + " " + history] = [0.5, 0.5]
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


if __name__ == '__main__':
    a = FictitiousPlay()
    for i in range(1, 2000):
        for player in range(2):
            startingReaches = [1, 1, 1]
            a.traverser = player
            a.walk_tree("", startingReaches)

        a.update_strategies(i)
        a.best_responses.clear()
    print(a.average_strategies)
