from breakout import BreakoutEnv
from agent import MonteCarloAgent
import matplotlib.pyplot as plt
import time

layouts = ["default", "rectangle", "triangle", "zigzag"]
num_episodes = 100000
training_times = {}

for layout in layouts:
    print(f"\nTraining on layout: {layout}")
    env = BreakoutEnv(layout_name=layout)
    agent = MonteCarloAgent(actions=[-1, 0, 1], epsilon=0.1, gamma=1.0)
    rewards_per_episode = []

    # Start timing
    start_time = time.time()

    for episode_num in range(num_episodes):
        state = env.reset()
        episode = []
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state

        agent.update(episode)
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        rewards_per_episode.append(total_reward)

        if episode_num % 1000 == 0:
            print(f"Episode {episode_num} finished for {layout}")

    # End timing
    end_time = time.time()
    duration = end_time - start_time
    training_times[layout] = duration
    print(f"Training time for {layout}: {duration:.2f} seconds")

    # Plot reward graph
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Reward Curve - {layout.capitalize()} Layout")
    plt.savefig(f"reward_{layout}.png")
    plt.close()

    print(f"Saved reward_{layout}.png")

# Print all times at the end
print("\n=== Training Times Summary ===")
for layout, t in training_times.items():
    print(f"{layout.capitalize()}: {t:.2f} seconds")
