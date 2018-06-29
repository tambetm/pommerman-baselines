import cpommerman
import pommerman
from pommerman.agents import BaseAgent
import numpy as np
from keras.models import load_model
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('model_file')
parser.add_argument('--num_episodes', type=int, default=10)
parser.add_argument('--render', action='store_true', default=False)
args = parser.parse_args()

# load pre-trained model
model = load_model(args.model_file)

env = cpommerman.make()
if args.render:
    # use original environment for rendering
    agents = [
        BaseAgent(),
        BaseAgent(),
        BaseAgent(),
        BaseAgent(),
    ]
    env_render = pommerman.make('PommeFFACompetition-v0', agents)

start_time = time.time()
steps = 0
for i in range(args.num_episodes):
    env.reset()
    done = False
    step = 0
    while not done:
        if args.render:
            # copy JSON state to original environment and render it
            state = env.get_json_info()
            env_render._init_game_state = state
            env_render.set_json_info()
            env_render.render()

        # use features to predict actions
        features = env.get_features()
        probs, _ = model.predict(features)
        actions = np.argmax(probs, axis=1).astype(np.uint8)

        # step environment, collect rewards and terminal condition
        env.step(actions)
        rewards = env.get_rewards()
        done = env.get_done()

        step += 1
        steps += 1
        print("Step:", step, "Actions:", actions, "Rewards:", rewards, "Done:", done)

elapsed = time.time() - start_time
print("Time:", elapsed, "Steps:", steps, "Time per step:", elapsed / steps, "FPS:", steps / elapsed)
