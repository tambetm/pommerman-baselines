import cpommerman
import numpy as np
import time

env = cpommerman.make()

start_time = time.time()
steps = 0
for i in range(1000):
    env.reset()
    done = False
    while not done:
        #state = env.get_state()
        #obs = env.get_observations()
        features = env.get_features()
        # use features, observations or state to produce action
        actions = np.random.randint(6, size=4, dtype=np.uint8)
        env.step(actions)
        rewards = env.get_rewards()
        done = env.get_done()
        steps += 1

elapsed = time.time() - start_time
print("Time:", elapsed, "Steps:", steps, "Time per step:", elapsed / steps, "FPS:", steps / elapsed)
