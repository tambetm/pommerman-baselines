import pommerman
from pommerman.agents import BaseAgent
from keras.models import load_model
import numpy as np
import time
import argparse


def featurize(obs):
    # TODO: history of n moves?
    board = obs['board']

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e.value for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    return np.stack(maps, axis=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    model = load_model(args.model_file)

    agents = [
        BaseAgent(),
        BaseAgent(),
        BaseAgent(),
        BaseAgent(),
    ]
    env = pommerman.make('PommeFFACompetition-v0', agents)

    total_rewards = 0
    total_lens = 0
    total_steps = 0
    start_time = time.time()
    for i in range(args.num_episodes):
        obs = env.reset()
        done = False
        step = 0
        lens = [None, None, None, None]
        while not done:
            if args.render:
                env.render()
            feats = np.array([featurize(o) for o in obs])
            probs, _ = model.predict(feats)
            actions = np.argmax(probs, axis=1)
            obs, rewards, done, info = env.step(actions)
            step += 1
            for i in range(4):
                if lens[i] is None and rewards[i] != 0:
                    lens[i] = step
            print("Step:", step, "Actions:", actions, "Rewards:", rewards, "Done:", done)

        total_rewards += sum(rewards)
        total_lens += sum(lens)
        total_steps += step

    elapsed = time.time() - start_time
    print("Average reward:", total_rewards / (args.num_episodes * 4), "Average length:", total_lens / (args.num_episodes * 4), "Time per timestep:", elapsed / total_steps)
