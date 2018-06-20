import pommerman
from pommerman import agents
import numpy as np
import time
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
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


class KerasAgent(agents.BaseAgent):
    def __init__(self, model_file):
        super().__init__()
        self.model = load_model(model_file)

    def act(self, obs, action_space):
        feat = featurize(obs)
        probs, values = self.model.predict(feat[np.newaxis])
        action = np.argmax(probs[0])
        #print("Action:", action)
        return action


def eval_model(agent_id, model_file, num_episodes):
    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]
    agent_list.insert(agent_id, KerasAgent(model_file))

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    rewards = []
    lengths = []
    start_time = time.time()
    # Run the episodes just like OpenAI Gym
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        lens = [None] * 4
        t = 0
        while not done:
            if args.render:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            for j in range(4):
                if lens[j] is None and reward[j] != 0:
                    lens[j] = t
            t += 1
        rewards.append(reward)
        lengths.append(lens)
        print('Episode ', i_episode, "reward:", reward[agent_id], "length:", lens[agent_id])
    elapsed = time.time() - start_time
    env.close()
    return rewards, lengths, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--num_episodes', type=int, default=400)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    assert args.num_episodes % 4 == 0, "The number of episodes should be divisible by 4"

    # make sure TF does not allocate all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    rewards0, lengths0, elapsed0 = eval_model(0, args.model_file, args.num_episodes // 4)
    rewards1, lengths1, elapsed1 = eval_model(1, args.model_file, args.num_episodes // 4)
    rewards2, lengths2, elapsed2 = eval_model(2, args.model_file, args.num_episodes // 4)
    rewards3, lengths3, elapsed3 = eval_model(3, args.model_file, args.num_episodes // 4)

    rewards = [(r0[0], r1[1], r2[2], r3[3]) for r0, r1, r2, r3 in zip(rewards0, rewards1, rewards2, rewards3)]
    lengths = [(l0[0], l1[1], l2[2], l3[3]) for l0, l1, l2, l3 in zip(lengths0, lengths1, lengths2, lengths3)]

    print("Average reward:", np.mean(rewards))
    print("Average length:", np.mean(lengths))

    print("Average rewards per position:", np.mean(rewards, axis=0))
    print("Average lengths per position:", np.mean(lengths, axis=0))

    elapsed = elapsed0 + elapsed1 + elapsed2 + elapsed3
    total_timesteps = np.sum(np.max(np.concatenate([lengths0, lengths1, lengths2, lengths3], axis=0), axis=1))
    print("Time per timestep:", elapsed / total_timesteps)
