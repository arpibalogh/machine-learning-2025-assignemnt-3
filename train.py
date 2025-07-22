from breakout import BreakoutEnv
from agent import MonteCarloAgent
import time
import matplotlib.pyplot as plt

layouts = ["default", "rectangle", "triangle", "zigzag"]
num_episodes = 10000
results = {}

for layout in layouts:
    print(f"\nTraining on layout: {layout}")
    env = BreakoutEnv(layout_name=layout)
    agent = MonteCarloAgent(actions=[-1, 0, 1], epsilon=0.1, gamma=1.0)

    rewards_per_episode = []

    for episode_num in range(num_episodes):
        state = env.reset()
        episode = []

        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state

        agent.update(episode)
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        rewards_per_episode.append(total_reward)

    results[layout] = rewards_per_episode

    print(f"Finished training on {layout}.\n")

# Plot reward curves
for layout in layouts:
    plt.plot(results[layout], label=layout)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Monte Carlo Training - Reward Comparison")
plt.legend()
plt.show()
