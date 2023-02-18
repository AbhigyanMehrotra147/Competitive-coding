import gym
import numpy as np
import random


class Frozen_lake:

    def __init__(self, learning_rate, discount_rate, epsilon, epsilon_decay_rate, sample_no_games, max_steps_size, min_epsilon,
                 max_epsilon) -> None:
        # Initializing hyper parameters
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.sample_no_games = sample_no_games
        self.max_step_size = max_steps_size
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

        # initializing the environment
        self.env = gym.make(id="FrozenLake-v1",
                            render_mode="rgb_array", is_slippery=False)
        self.No_actions = self.env.action_space.n
        self.No_states = self.env.observation_space.n

        # Initializing variables we need to track to the game
        self.score = 0
        self.total_rewards = 0
        self.Q_table = np.zeros(shape=(self.No_states, self.No_actions))
        self._Q_table_over_time = np.zeros(
            shape=(self.sample_no_games, self.No_states, self.No_actions))
    # This function updates the Q table

    def update_q_table(self, curr_state, action, new_state, reward) -> np.ndarray:
        max_new_state_value = np.max(self.Q_table[new_state])
        self.Q_table[curr_state][action] += self.learning_rate * \
            (reward + self.discount_rate*max_new_state_value -
             self.Q_table[curr_state][action])

    def update_epsilon(self, game_no):
        self.epsilon = self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon) * \
            np.exp(-self.epsilon_decay_rate*game_no)

    # Function chooses action based on the epsilon value
    def choose_action(self, curr_state):
        if random.random() > self.epsilon:
            return np.argmax(self.Q_table[curr_state])
        else:
            return self.env.action_space.sample()

    # This is the iterative function which trains the model
    def run(self):

        for game_no in range(self.sample_no_games):
            curr_state, _ = self.env.reset()
            terminate = False
            reward = 0
            step = 0
            for step in range(self.max_step_size):
                action = self.choose_action(curr_state=curr_state)
                new_state, reward, terminate, truncate, info = self.env.step(
                    action=action)
                # updating Q_table here
                self.update_q_table(
                    curr_state=curr_state, action=action, new_state=new_state, reward=reward)
                curr_state = new_state
                if(terminate):
                    break
            self.update_epsilon(game_no=game_no)
            self.total_rewards += reward
            self._Q_table_over_time[game_no] = self.Q_table

        # Returning the score
        self.env.close()
        print(self.Q_table)
        return self.total_rewards/self.sample_no_games

    # Function tries out the result of the game
    def try_out(self):
        for i in range(10):
            self.env = gym.make(id="FrozenLake-v1",
                                render_mode="human", is_slippery=False)
            state, _ = self.env.reset()
            terminated = False
            while(not terminated):
                action = np.argmax(self.Q_table[state])
                state, reward, terminated, truncated, info = self.env.step(
                    action)


sample_no_games = 5000
learning_rate = 0.9
discount_rate = 0.95
max_step_size = 100
epsioon_decay_rate = 0.001
epsilon = 1
min_epsilon = 0.01
max_epsilon = 1

F_L = Frozen_lake(learning_rate=learning_rate, discount_rate=discount_rate, epsilon=epsilon, epsilon_decay_rate=epsioon_decay_rate,
                  sample_no_games=sample_no_games, max_steps_size=max_step_size, min_epsilon=min_epsilon, max_epsilon=max_epsilon)
print(F_L.run())
print(F_L.Q_table)
