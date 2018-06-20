import pommerman
from pommerman import agents
import numpy as np
import argparse
from copy import deepcopy


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--render', action="store_true", default=False)
    parser.add_argument('out_file')
    args = parser.parse_args()

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    observations = [[], [], [], []]
    actions = [[], [], [], []]
    rewards = [[], [], [], []]

    # Run the episodes just like OpenAI Gym
    for i in range(args.num_episodes):
        obs = env.reset()
        done = False
        reward = [0, 0, 0, 0]
        t = 0
        while not done:
            if args.render:
                env.render()
            action = env.act(obs)
            new_obs, new_reward, done, info = env.step(action)
            for j in range(4):
                if reward[j] == 0:
                    observations[j].append(featurize(obs[j]))
                    actions[j].append(action[j])
                    rewards[j].append(new_reward[j])
            obs = deepcopy(new_obs)
            reward = deepcopy(new_reward)
            t += 1
        print("Episode:", i + 1, "Max length:", t, "Rewards:", reward)
    env.close()

    np.savez_compressed(args.out_file,
                        observations=sum(observations, []),
                        actions=sum(actions, []),
                        rewards=sum(rewards, []))
