from breakout import BreakoutEnv
from agent import MonteCarloAgent
import random
import time

env = BreakoutEnv()
agent = MonteCarloAgent(actions=[-1, 0, 1], epsilon=0.1, gamma=1.0)

num_episodes = 100000

for episode_num in range(num_episodes):
    state = env.reset()
    episode = []

    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    agent.update(episode)

    if episode_num % 1000 == 0:
        print(f"Episode {episode_num} finished")

# Testing
env = BreakoutEnv()
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    time.sleep(0.1)
    action = agent.choose_action(state)
    state, reward, done = env.step(action)
    total_reward += reward

print("Final reward:", total_reward)

