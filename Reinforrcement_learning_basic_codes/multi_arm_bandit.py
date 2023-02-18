import matplotlib.pyplot as plt
import random
import numpy as np

"""
In this file code is writted to experiment with different values of epsilon for the multi-arm bandit problem
here we assume that true value of every action is stationary.) The goal is to make a program which can maximize
rewards.
"""


class Multi_arm_badit:
    """
    The meaning of all class members is:
    1. self.increment_rate_epsilon is the rate at which epsilon is to be incremented after every self.no_iterations
    2. self.epsilon is the current epsilon value
    3. self.action_values is the true value of the actions
    4. self.bandit_estimates_action_value is the estimate of the value of each action at a particular iteration
    5. self.no_of_action is the number of actions the bandit can choose from
    6. self.no_iterations is the number of iterations for the bandit must maximixe his value function
    7. self.max_value is the max value of self.bandit_estimated_action_value
    8. self.max_index is the index of max_value in self.bandit_estimated_action_value  
    9. self.action_stds is the the stds in the reward that each action provides
    10. self.No_of_times_action_chosen keeps the count of how many times each action is chosen
    11. self.rewards keeps track of the all rewards for one iteration cycle
    """

    def __init__(self, init_rate, init_epsilon, true_action_values, stds_of_actions,  no_iteration, no_actions) -> None:

        self.increment_rate_epsilon = init_rate
        self.epsilon = init_epsilon

        self.action_values = true_action_values
        self.action_stds = stds_of_actions
        self.bandit_estimed_action_value = np.zeros(shape=(no_actions, 1))
        self.No_times_action_chosen = np.zeros(shape=(no_actions, 1))

        self.no_actions = no_actions
        self.no_iterations = no_iteration

        self.max_value = 0
        self.max_index = 0

        self.rewards = None
    # Generates a Random number from a range of values

    # step = False means only integer values will be returned
    @ staticmethod
    def Generate_random(start, stop, step=False) -> int:
        if(not step):
            return random.randrange(start=start, stop=stop)
        return round(random.uniform(a=start, b=stop), 2)

    # Depending on the self.increment_rate_epsilon the epsilon value is updated
    def update_epsilon_value(self) -> None:
        self.epsilon = self.epsilon + self.increment_rate_epsilon

    # Depending on the way we are selecting greedy action the function chooses the greedy action
    def choose_action_to_exploit(self) -> None:
        for i in range(self.no_actions):
            value = self.bandit_estimed_action_value[i]
            if(value > self.max_value):
                self.max_value = value
                self.max_index = i
        # print(self.max_index)
        return self.max_index

    # Gives the index of the action to be explored
    def choose_action_to_explore(self) -> int:
        action = self.max_index
        while(action == self.max_index):
            action = self.Generate_random(
                start=0, stop=self.no_actions, step=False)

        return action

    # function gets the reward from action by considering true mean reward value and std
    # Also updates the number of times the action was chosen
    def get_reward(self, chosen_action):
        self.No_times_action_chosen[chosen_action] += 1
        return self.Generate_random(start=self.action_values[chosen_action] - self.action_stds[chosen_action],
                                    stop=self.action_values[chosen_action] -
                                    self.action_stds[chosen_action],
                                    step=True)

    # the function updates the estimated action value after each iteration
    def update_action_value(self, chosen_action, reward) -> None:
        self.bandit_estimed_action_value[chosen_action] = self.bandit_estimed_action_value[chosen_action] + (1/self.No_times_action_chosen[chosen_action])\
            * (reward - self.bandit_estimed_action_value[chosen_action])
    # Runs a loop for self.iteration to see the convergence to the optimal reward
    # Also fills the table for storing rewards for different epsilons.

    def run(self) -> None:
        Average_reward = []
        epsilons = []
        exploration = False
        while (self.epsilon < 0.3):
            j = 1
            self.rewards = np.zeros(shape=(self.no_iterations, 1))
            chosen_action = None
            temp_epsilon = self.epsilon
            for i in range(self.no_iterations):
                if(self.Generate_random(start=0, stop=1, step=True) < temp_epsilon):
                    j += 1
                    exploration = True
                if(exploration):
                    chosen_action = self.choose_action_to_explore()
                if(not exploration):
                    chosen_action = self.choose_action_to_exploit()

                reward = self.get_reward(chosen_action=chosen_action)

                self.rewards[i] = reward
                self.update_action_value(
                    chosen_action=chosen_action, reward=reward)
                exploration = False
                temp_epsilon -= 0.0002

            self.plot()
            self.update_epsilon_value()
            Average_reward.append(np.mean(self.rewards))
            epsilons.append(self.epsilon)
        # plt.plot(epsilons, Average_reward)
        # plt.show()

    def plot(self):
        plt.plot(np.arange(self.no_iterations), self.rewards)
        plt.xlim(0, self.no_iterations)
        plt.ylim(-10, np.max(self.action_values) + 2)
        plt.xlabel("iteration number")
        plt.ylabel("reward value")

        plt.show()


init_rate = 0.01
init_epsilon = 0.01
no_iterations = 1000
no_actions = 10
action_values = np.array([0, 1, 2, -4, 5, 10, 1, -1.5, -2, -9])
std_values = np.array([0.5, 0.2, 0.3, 0.1, 0.5, 0.05, 0.4, 0.2, 0.43, 0.12])


M = Multi_arm_badit(init_rate=init_rate, init_epsilon=init_epsilon, no_actions=no_actions, no_iteration=no_iterations,
                    true_action_values=action_values, stds_of_actions=std_values)
M.run()
M.plot()
