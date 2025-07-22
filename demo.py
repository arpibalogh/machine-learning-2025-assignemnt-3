from breakout import BreakoutEnv
from agent import MonteCarloAgent
import time

# Create a trained agent (in real use, you'd want to load one — this just demos behavior)
agent = MonteCarloAgent(actions=[-1, 0, 1])

# List of layouts to demo
layouts = ["default", "rectangle", "triangle", "zigzag"]

for layout in layouts:
    print(f"\nShowing gameplay for layout: {layout}")
    env = BreakoutEnv(layout_name=layout)
    state = env.reset()
    done = False

    time.sleep(1)
    steps = 0

    while not done and steps < 100:
        env.render()
        time.sleep(0.2)
        action = agent.choose_action(state)  # Just picks based on Q-values (random at start)
        state, reward, done = env.step(action)
        steps += 1

    print(f"Finished layout: {layout}")
    input("Press Enter to continue to next layout...")
