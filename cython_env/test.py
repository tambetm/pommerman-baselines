import pommerman
from pommerman import agents
import cpommerman
import numpy as np
import time
import argparse


# featurization code that applies to old environment observations
def featurize_old(obs):
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


# featurization code that applies to new environment observations
def featurize_new(obs):
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
        maps.append(board == obs['teammate'])
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    return np.stack(maps, axis=2)


parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=1)
parser.add_argument('--render', action="store_true", default=False)
args = parser.parse_args()

# set up two environments, agents are only in old env
agents = [
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
    agents.SimpleAgent(),
]
env_old = pommerman.make('PommeFFACompetition-v0', agents)
env_new = cpommerman.make()

total_time_old = 0
total_time_new = 0
total_state_len = 0
max_state_len = 0

n = 0
for i in range(args.num_episodes):
    obs_old = env_old.reset()
    # just to test for errors in reset code,
    # impossible to verify because random seed is not implemented in both envs
    obs_new = env_new.reset()
    done_old = False
    t = 0
    while not done_old:
        if args.render:
            env_old.render()

        # TEST 1: verify set_json_info() and get_json_info()
        # copy old env state to new env
        state_old = env_old.get_json_info()
        env_new.set_json_info(state_old)
        state_new = env_new.get_json_info()
        # item order is not consistent
        state_old['items'] = list(sorted(state_old['items']))
        state_new['items'] = list(sorted(state_new['items']))
        # verify that JSON state of new environment matches old environment
        assert str(state_old) == str(state_new), "\n" + str(state_old) + "\n\n" + str(state_new)

        # TEST 2: verify binary state
        state_before = env_new.get_state()
        env_new.set_state(state_before)
        state_after = env_new.get_state()
        # verify that state did not change as a result of set_state()
        assert state_before == state_after, "\n" + str(state_before) + "\n\n" + str(state_after)
        total_state_len += len(state_before)
        if len(state_before) > max_state_len:
            max_state_len = len(state_before)

        # TEST 3: verify that binary state matches json
        state_new = env_new.get_json_info()
        state_new['items'] = list(sorted(state_new['items']))
        # verify that new JSON state matches previous JSON state
        assert str(state_old) == str(state_new), "\n" + str(state_old) + "\n\n" + str(state_new)

        # take action using SimpleAgents
        actions = env_old.act(obs_old)
        # have to convert actions to uint8 (unsigned char)
        actions = np.array(actions, dtype=np.uint8)

        # time old environment step
        start_time = time.time()
        obs_old, reward_old, done_old, info_old = env_old.step(actions)
        # include featurization in step, because it is essential
        feat_old = np.array([featurize_old(o) for o in obs_old])
        old_time = time.time() - start_time
        total_time_old += old_time

        # time new environment step
        start_time = time.time()
        env_new.step(actions)
        feat_new = env_new.get_features()
        reward_new = env_new.get_rewards()
        done_new = env_new.get_done()
        #info_new = env_new.get_info()
        new_time = time.time() - start_time
        total_time_new += new_time

        # TEST 4: verify that fast feature generation generates the same features
        obs_new = env_new.get_observations()
        feat_new_old = np.array([featurize_new(o) for o in obs_new])
        for agent_id in range(4):
            for channel in range(18):
                assert np.all(feat_new_old[agent_id, :, :, channel] == feat_new[agent_id, :, :, channel]), \
                    "Agent %d channel %d:\n" % (agent_id, channel) + str(feat_new_old[agent_id, :, :, channel]) + "\n\n" + str(feat_new[agent_id, :, :, channel])

        t += 1
        n += 1
        print("Step: ", t, "Actions:", actions, "Rewards:", reward_old, "Done:", done_old, "Old time:", old_time, "New time:", new_time)

        try:
            # TEST 5: verify that features from new environment match features from old
            # features are easier to compare than observations and essentially contain the same information
            for agent_id in range(4):
                assert np.all(feat_old[agent_id] == feat_new[agent_id]), "Agent %d observations:\n" % agent_id + str(obs_old[agent_id]) + "\n\n" + str(obs_new[agent_id])
            # TEST 6: verify that rewards from both environments match
            assert np.all(np.array(reward_old) == np.array(reward_new)), str(reward_old) + " != " + str(reward_new)
            # TEST 7: verify that done flag from both environments matches
            assert done_old == done_new, str(done_old) + " != " + str(done_new)
        except AssertionError:
            import traceback
            traceback.print_exc()
            input()

        # TEST 8: verify that setting state actually works
        # set state to previous timestep and compare that JSON matches
        env_new.set_state(state_before)
        state_new = env_new.get_json_info()
        state_new['items'] = list(sorted(state_new['items']))
        assert str(state_old) == str(state_new), "\n" + str(state_old) + "\n\n" + str(state_new)

print("Original fps:", n / total_time_old, "New fps:", n / total_time_new, "Speedup:", total_time_old / total_time_new)
print("Avg state len:", total_state_len / n, "Max state len:", max_state_len)
