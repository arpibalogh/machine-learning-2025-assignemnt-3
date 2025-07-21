from breakout import BreakoutEnv
import random
import time

env = BreakoutEnv()
state = env.reset()

done = False
while not done:
    env.render()
    time.sleep(0.2)
    action = random.choice([-1, 0, 1])
    state, reward, done = env.step(action)
