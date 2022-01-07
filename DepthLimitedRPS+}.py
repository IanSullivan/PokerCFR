import numpy as np
from numpy.random import choice


class RPSTrainer:
    def __init__(self):
        self.NUM_ACTIONS = 3
        self.possible_actions = np.arange(self.NUM_ACTIONS)

        # Rewards for rock paper scissors plus
        self.actionUtility = np.array([
            [0, -1, 2],
            [1, 0, -2],
            [-2, 2, 0]
        ])

        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.strategySum = np.zeros(self.NUM_ACTIONS)

        self.opp_regret_strategies = [[0, 0, 0]]
        self.opp_regret_sum = np.zeros(len(self.opp_regret_strategies))
        self.opp_strategy_sum = np.zeros(len(self.opp_regret_strategies))

    def reset_regrets(self):
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.opp_regret_sum = np.zeros(len(self.opp_regret_strategies))
        self.opp_strategy_sum = np.zeros(len(self.opp_regret_strategies))

    @staticmethod
    def get_strategy(regret_sum):
        # Clip negative regrets for faster convergence
        regret_sum = np.clip(regret_sum, a_min=0, a_max=None)
        normalizing_sum = np.sum(regret_sum)
        strategy = regret_sum
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.repeat(1/len(regret_sum), len(regret_sum))

        return strategy

    def get_average_strategy(self, strategy_sum):
        average_strategy = [0, 0, 0]
        normalizing_sum = sum(strategy_sum)
        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                average_strategy[a] = strategy_sum[a] / normalizing_sum
            else:
                average_strategy[a] = 1.0 / self.NUM_ACTIONS
        return average_strategy

    @staticmethod
    def get_action(strategy):
        return choice(np.arange(len(strategy)), p=strategy)

    def get_reward(self, my_action, opponent_action):
        return self.actionUtility[my_action, opponent_action]

    def opposition_strategy(self, iterations, hero_strategy):
        values = [0 for _ in range(self.NUM_ACTIONS)]
        for i in range(iterations):
            hero_choice = choice(self.possible_actions, p=hero_strategy)
            villain_choice = choice(self.possible_actions)
            values[villain_choice] += self.get_reward(villain_choice, hero_choice)
        policy = self.get_strategy(np.array(values, dtype=np.float64))
        return policy

    def calc_EV(self, p2_strats):
        ev = [0 for _ in range(self.NUM_ACTIONS)]
        for i in range(self.NUM_ACTIONS):
            for j, p2_strategy in enumerate(p2_strats):
                ev[i] += p2_strategy * self.get_reward(i, j)
        return ev

    def train(self, iterations):
        for i in range(iterations):
            strategy = self.get_strategy(self.regretSum)
            opp_strategy = self.get_strategy(self.opp_regret_sum)
            self.strategySum += strategy
            self.opp_strategy_sum += opp_strategy

            opponent_action = self.get_action(opp_strategy)
            my_action = self.get_action(strategy)

            my_reward = self.opp_regret_strategies[opponent_action][my_action]
            opp_reward = -1 * self.opp_regret_strategies[opponent_action][my_action]

            # Calculate regret for each of the strategies
            for a in range(len(self.opp_regret_strategies)):
                self.opp_regret_sum[a] += (-1 * self.opp_regret_strategies[a][my_action]) - opp_reward

            for a in range(self.NUM_ACTIONS):
                self.regretSum[a] += self.opp_regret_strategies[opponent_action][a] - my_reward


def main():
    trainer = RPSTrainer()
    for i in range(10):
        # Solve current sub game
        trainer.train(5000)
        target_policy = trainer.get_average_strategy(trainer.strategySum)
        # Calculate best response
        new_policy = trainer.opposition_strategy(5000, target_policy)
        new_policy_ev = trainer.calc_EV(new_policy)
        # Add best response to leaf node policies
        trainer.opp_regret_strategies.append(new_policy_ev)
        print('Target policy: %s' % (target_policy))
        trainer.reset_regrets()

    print('Final policy: %s' % (target_policy))


if __name__ == "__main__":
    main()
