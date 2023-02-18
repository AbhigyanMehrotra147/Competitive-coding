import gym
import numpy as np
import random
# env = gym.make("FrozenLake-v1", render_mode="human")
# env.reset()

# env.render()
# print(env.observation_space)
# print(env.action_space)

Quality = np.zeros(shape=(16, 4))
sample_no_games = 10000
learning_rate = 0.2
gamma = 0.9
epsilon = 0.1
Total_reward = 0

env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
while(sample_no_games):

    env.reset()
    sample_no_steps = 500
    prev_state = 0
    terminated = False
    epsilon = 0.1
    while(sample_no_steps and not terminated):

        if random.random() < epsilon:
            action = random.randint(a=0, b=3)
        else:
            action = np.argmax(Quality[prev_state, :])

        state, reward, terminated, truncated, info = env.step(action)
        Quality[prev_state][action] += learning_rate * \
            (reward + gamma *
             np.max(Quality[state, :]) - Quality[prev_state][action])

        prev_state = state
        sample_no_steps -= 1
        # print(sample_no_steps)
        epsilon *= 0.999
        Total_reward += reward
    sample_no_games -= 1

env.close()
env = gym.make("FrozenLake-v1", render_mode="human")
print(Quality)
terminated = False
env.reset()
state = 0
print(Total_reward)
while(True):
    action = np.argmax(Quality[state, :])
    state, reward, terminated, truncated, info = env.step(action)
