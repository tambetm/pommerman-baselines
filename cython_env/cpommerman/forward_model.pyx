import numpy as np
cimport numpy as np

cimport cython
cimport constants
cimport characters
cimport utility
from utility cimport Position, Position_eq, Position_neq, byte

from libc.string cimport memset
from libc.string cimport memcpy


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef byte explode(byte r, byte c, byte[constants.BOARD_SIZE][constants.BOARD_SIZE] exploded_map, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board):
    if not (r >= 0 and c >= 0 and r < constants.BOARD_SIZE and c < constants.BOARD_SIZE):
        return False
    if board[r][c] == constants.Item.Rigid:
        return False
    exploded_map[r][c] = 1
    if board[r][c] == constants.Item.Wood:
        return False
    return True


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef tuple make_bomb_maps(Position position, list bombs, byte is_partially_observable, byte agent_view_size):
    cdef characters.Bomb bomb
    cdef np.ndarray blast_strengths_np, life_np
    cdef unsigned char[:, :] blast_strengths, life

    blast_strengths = blast_strengths_np = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE), dtype=np.uint8)
    life = life_np = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE), dtype=np.uint8)

    for bomb in bombs:
        if not is_partially_observable \
           or in_view_range(position, bomb.position.row, bomb.position.col, agent_view_size):
            blast_strengths[bomb.position.row, bomb.position.col] = bomb.blast_strength
            life[bomb.position.row, bomb.position.col] = bomb.life
    return blast_strengths_np, life_np

cdef byte in_view_range(Position position, byte row, byte col, byte agent_view_size):
    return position.row >= row - agent_view_size and position.row <= row + agent_view_size \
        and position.col >= col - agent_view_size and position.col <= col + agent_view_size

cdef byte any_lst_equal(list lst, list values):
    for v in values:
        if lst == v:
            return True
    return False


cdef class ForwardModel(object):
    """Class for helping with the [forward] modeling of the game state."""
    '''
    cdef list act(self, list agents, list obs, object action_space, byte is_communicative=False):
        """Returns actions for each agent in this list.

        Args:
          agents: A list of agent objects.
          obs: A list of matching observations per agent.
          action_space: The action space for the environment using this model.
          is_communicative: Whether the action depends on communication
            observations as well.

        Returns a list of actions.
        """

        def act_ex_communication(agent):
            if agent.is_alive:
                return agent.act(obs[agent.agent_id], action_space=action_space)
            else:
                return constants.Action.Stop

        def act_with_communication(agent):
            if agent.is_alive:
                action = agent.act(
                    obs[agent.agent_id], action_space=action_space)
                if type(action) == int:
                    action = [action] + [0, 0]
                assert (type(action) == list)
                return action
            else:
                return [constants.Action.Stop, 0, 0]

        ret = []
        for agent in agents:
            if is_communicative:
                ret.append(act_with_communication(agent))
            else:
                ret.append(act_ex_communication(agent))
        return ret
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cdef void step(self, const unsigned char[:] actions,
                   byte[constants.BOARD_SIZE][constants.BOARD_SIZE] curr_board,
                   byte[constants.BOARD_SIZE][constants.BOARD_SIZE] curr_items,
                   list curr_agents,
                   list curr_bombs,
                   list curr_flames,
                   byte max_blast_strength=10):
        cdef characters.Bomber agent, agent2
        cdef characters.Bomb bomb, bomb2
        cdef characters.Flame flame
        cdef Position position, desired_position, curr_position, target_position, agent_position, bomb_position
        cdef constants.Action action, direction
        cdef list alive_agents
        cdef byte board_size, num_agent, num_agent2, num_bomb, num, bomb_occupancy_, agent_occupancy_, r, c, i
        cdef byte isAgent, change, has_new_explosions
        cdef constants.Item item_value
        # NB! fixed board size!
        cdef byte[constants.BOARD_SIZE][constants.BOARD_SIZE] agent_occupancy, bomb_occupancy, exploded_map
        cdef signed char[constants.BOARD_SIZE][constants.BOARD_SIZE][2] crossings

        # initialize local arrays with zeros
        memset(agent_occupancy, 0, sizeof(agent_occupancy))
        memset(bomb_occupancy, 0, sizeof(bomb_occupancy))
        memset(exploded_map, 0, sizeof(exploded_map))
        memset(crossings, 0, sizeof(crossings))

        # Tick the flames. Replace any dead ones with passages. If there is an
        # item there, then reveal that item.
        # Iterate over copy of the list so we can remove elements inside the loop.
        for flame in curr_flames[:]:
            position = flame.position
            if flame.is_dead():
                curr_flames.remove(flame)
                item_value = <constants.Item>curr_items[position.row][position.col]
                if item_value:
                    curr_items[position.row][position.col] = 0
                else:
                    item_value = constants.Item.Passage
                curr_board[position.row][position.col] = item_value
            else:
                flame.tick()

        # Step the living agents and moving bombs.
        # If two agents try to go to the same spot, they should bounce back to
        # their previous spots. This is complicated with one example being when
        # there are three agents all in a row. If the one in the middle tries
        # to go to the left and bounces with the one on the left, and then the
        # one on the right tried to go to the middle one's position, she should
        # also bounce. A way of doing this is to gather all the new positions
        # before taking any actions. Then, if there are disputes, correct those
        # disputes iteratively.
        # Additionally, if two agents try to switch spots by moving into each
        # Figure out desired next position for alive agents
        alive_agents = [agent for agent in curr_agents if agent.is_alive]
        for agent in alive_agents:
            agent.desired_position = agent.position
            agent.delayed_position = Position(-1, -1)
            agent.kicked_bomb = None

        for agent in alive_agents:
            position = agent.position
            # We change the curr_board here as a safeguard. We will later
            # update the agent's new position.
            curr_board[position.row][position.col] = constants.Item.Passage
            action = <constants.Action>actions[agent.agent_id]

            if action == constants.Action.Stop:
                pass
            elif action == constants.Action.LayBomb:
                position = agent.position
                if not utility.position_is_bomb(curr_bombs, position):
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
            elif utility.is_valid_direction(curr_board, position, action):
                agent.desired_position = agent.get_next_position(action)

        # Gather desired next positions for moving bombs. Handle kicks later.
        for bomb in curr_bombs:
            bomb.desired_position = bomb.position
            bomb.delayed_position = Position(-1, -1)
            bomb.kicked_agent = None

        for bomb in curr_bombs:
            curr_board[bomb.position.row][bomb.position.col] = constants.Item.Passage
            if bomb.is_moving():
                desired_position = utility.get_next_position(
                    bomb.position, bomb.moving_direction)
                if utility.position_on_board(curr_board, desired_position) \
                    and not utility.position_is_powerup(curr_board, desired_position) \
                        and not utility.position_is_wall(curr_board, desired_position):
                    bomb.desired_position = desired_position

        # Position switches:
        # Agent <-> Agent => revert both to previous position.
        # Bomb <-> Bomb => revert both to previous position.
        # Agent <-> Bomb => revert Bomb to previous position.
        for num_agent, agent in enumerate(alive_agents):
            desired_position = agent.desired_position
            curr_position = agent.position
            if Position_neq(desired_position, curr_position):
                if curr_position.row != desired_position.row:
                    assert curr_position.col == desired_position.col
                    r = min(curr_position.row, desired_position.row)
                    c = curr_position.col
                    i = 0
                else:
                    assert curr_position.row == desired_position.row
                    r = curr_position.row
                    c = min(curr_position.col, desired_position.col)
                    i = 1
                num_agent2 = crossings[r][c][i]
                if num_agent2 != 0:
                    # Crossed another agent - revert both to prior positions.
                    agent.desired_position = agent.position
                    agent2 = alive_agents[num_agent2 - 1]
                    agent2.desired_position = agent2.position
                else:
                    crossings[r][c][i] = num_agent + 1

        for num_bomb, bomb in enumerate(curr_bombs):
            desired_position = bomb.desired_position
            curr_position = bomb.position
            if Position_neq(desired_position, curr_position):
                if curr_position.row != desired_position.row:
                    assert curr_position.col == desired_position.col
                    r = min(curr_position.row, desired_position.row)
                    c = curr_position.col
                    i = 0
                else:
                    assert curr_position.row == desired_position.row
                    r = curr_position.row
                    c = min(curr_position.col, desired_position.col)
                    i = 1
                num = crossings[r][c][i]
                if num != 0:
                    # Crossed - revert to prior position.
                    bomb.desired_position = bomb.position
                    if num < 0:
                        # Crossed bomb - revert that to prior position as well.
                        bomb2 = curr_bombs[-num - 1]
                        bomb2.desired_position = bomb2.position
                else:
                    crossings[r][c][i] = -(num_bomb + 1)

        # Deal with multiple agents or multiple bomb collisions on desired next
        # position by resetting desired position to current position for
        # everyone involved in the collision.
        for agent in alive_agents:
            agent_occupancy[agent.desired_position.row][agent.desired_position.col] += 1
        for bomb in curr_bombs:
            bomb_occupancy[bomb.desired_position.row][bomb.desired_position.col] += 1

        # Resolve >=2 agents or >=2 bombs trying to occupy the same space.
        change = True
        while change:
            change = False
            for agent in alive_agents:
                desired_position = agent.desired_position
                curr_position = agent.position
                # Either another agent is going to this position or more than
                # one bomb is going to this position. In both scenarios, revert
                # to the original position.
                if Position_neq(desired_position, curr_position) and \
                        (agent_occupancy[desired_position.row][desired_position.col] > 1 or bomb_occupancy[desired_position.row][desired_position.col] > 1):
                    agent.desired_position = curr_position
                    agent_occupancy[curr_position.row][curr_position.col] += 1
                    change = True

            for bomb in curr_bombs:
                desired_position = bomb.desired_position
                curr_position = bomb.position
                if Position_neq(desired_position, curr_position) and \
                        (bomb_occupancy[desired_position.row][desired_position.col] > 1 or agent_occupancy[desired_position.row][desired_position.col] > 1):
                    bomb.desired_position = curr_position
                    bomb_occupancy[curr_position.row][desired_position.col] += 1
                    change = True

        # Loop through all bombs to see if they need a good kicking or cause
        # collisions with an agent.
        for bomb in curr_bombs:
            desired_position = bomb.desired_position

            if agent_occupancy[desired_position.row][desired_position.col] == 0:
                # There was never an agent around to kick or collide.
                continue

            for agent in alive_agents:
                if Position_eq(desired_position, agent.desired_position):
                    break
            else:
                continue

            if Position_eq(desired_position, agent.position):
                # Agent did not move
                if Position_neq(desired_position, bomb.position):
                    # Bomb moved, but agent did not. The bomb should revert
                    # and stop.
                    bomb.delayed_position = bomb.position
                continue

            # NOTE: At this point, we have that the agent in question tried to
            # move into this position.
            if not agent.can_kick:
                # If we move the agent at this point, then we risk having two
                # agents on a square in future iterations of the loop. So we
                # push this change to the next stage instead.
                bomb.delayed_position = bomb.position
                agent.delayed_position = agent.position
                continue

            # Agent moved and can kick - see if the target for the kick never had anyhing on it
            direction = <constants.Action>actions[agent.agent_id]
            target_position = utility.get_next_position(desired_position,
                                                        direction)
            if utility.position_on_board(curr_board, target_position) and \
                    agent_occupancy[target_position.row][target_position.col] == 0 and \
                    bomb_occupancy[target_position.row][target_position.col] == 0 and \
                    not utility.position_is_powerup(curr_board, target_position) and \
                    not utility.position_is_wall(curr_board, target_position):
                # Ok to update bomb desired location as we won't iterate over it again here
                # but we can not update bomb_occupancy on target position and need to check it again
                # However we need to set the bomb count on the current position to zero so
                # that the agent can stay on this position.
                bomb_occupancy[desired_position.row][desired_position.col] = 0
                bomb.delayed_position = target_position
                bomb.kicked_agent = agent
                agent.kicked_bomb = bomb
                bomb.moving_direction = direction
                # Bombs may still collide and we then need to reverse bomb and agent ..
            else:
                bomb.delayed_position = bomb.position
                agent.delayed_position = agent.position

        for bomb in curr_bombs:
            if Position_neq(bomb.delayed_position, Position(-1, -1)):
                bomb.desired_position = bomb.delayed_position
                bomb_occupancy[bomb.delayed_position.row][bomb.delayed_position.col] += 1
                change = True

        for agent in alive_agents:
            if Position_neq(agent.delayed_position, Position(-1, -1)):
                agent.desired_position = agent.delayed_position
                agent_occupancy[agent.delayed_position.row][agent.delayed_position.col] += 1
                change = True

        while change:
            change = False
            for agent in alive_agents:
                desired_position = agent.desired_position
                curr_position = agent.position
                # Agents and bombs can only share a square if they are both in their
                # original position (Agent dropped bomb and has not moved)
                if Position_neq(desired_position, curr_position) and \
                        (agent_occupancy[desired_position.row][desired_position.col] > 1 or
                            bomb_occupancy[desired_position.row][desired_position.col] != 0):
                    # Late collisions resulting from failed kicks force this agent to stay at the
                    # original position. Check if this agent successfully kicked a bomb above and undo
                    # the kick.
                    if agent.kicked_bomb is not None:
                        bomb = agent.kicked_bomb
                        bomb.desired_position = bomb.position
                        bomb_occupancy[bomb.position.row][bomb.position.col] += 1
                        bomb.kicked_agent = None
                        agent.kicked_bomb = None
                    agent.desired_position = curr_position
                    agent_occupancy[curr_position.row][curr_position.col] += 1
                    change = True

            for bomb in curr_bombs:
                desired_position = bomb.desired_position
                curr_position = bomb.position

                # This bomb may be a boomerang, i.e. it was kicked back to the
                # original location it moved from. If it is blocked now, it
                # can't be kicked and the agent needs to move back to stay
                # consistent with other movements.
                if Position_eq(desired_position, curr_position) and bomb.kicked_agent is None:
                    continue

                bomb_occupancy_ = bomb_occupancy[desired_position.row][desired_position.col]
                agent_occupancy_ = agent_occupancy[desired_position.row][desired_position.col]
                # Agents and bombs can only share a square if they are both in their
                # original position (Agent dropped bomb and has not moved)
                if bomb_occupancy_ > 1 or agent_occupancy_ != 0:
                    bomb.desired_position = curr_position
                    bomb_occupancy[curr_position.row][curr_position.col] += 1
                    if bomb.kicked_agent is not None:
                        agent = bomb.kicked_agent
                        agent.desired_position = agent.position
                        agent_occupancy[agent.position.row][agent.position.col] += 1
                        agent.kicked_bomb = None
                        bomb.kicked_agent = None
                    change = True

        for bomb in curr_bombs:
            if Position_eq(bomb.desired_position, bomb.position) and \
               bomb.kicked_agent is None:
                # Bomb was not kicked this turn and its desired position is its
                # current location. Stop it just in case it was moving before.
                bomb.stop()
            else:
                # Move bomb to the new position.
                # NOTE: We already set the moving direction up above.
                bomb.position = bomb.desired_position

        for agent in alive_agents:
            if Position_neq(agent.desired_position, agent.position):
                agent.move(<constants.Action>actions[agent.agent_id])
                if utility.position_is_powerup(curr_board, agent.position):
                    agent.pick_up(
                        <constants.Item>curr_board[agent.position.row][agent.position.col],
                        max_blast_strength=max_blast_strength)

        # Explode bombs.
        has_new_explosions = False

        for bomb in curr_bombs:
            bomb.tick()
            if bomb.exploded():
                has_new_explosions = True
            elif curr_board[bomb.position.row][bomb.position.col] == constants.Item.Flames:
                bomb.fire()
                has_new_explosions = True

        # Chain the explosions.
        while has_new_explosions:
            has_new_explosions = False
            # iterate over copy of the list, so we can remove elements inside the loop
            for bomb in curr_bombs[:]:
                if not bomb.exploded():
                    continue

                curr_bombs.remove(bomb)
                bomb.bomber.incr_ammo()
                for i in range(1, bomb.blast_strength):
                    if not explode(bomb.position.row - i, bomb.position.col, exploded_map, curr_board):
                        break
                for i in range(bomb.blast_strength):
                    if not explode(bomb.position.row + i, bomb.position.col, exploded_map, curr_board):
                        break
                for i in range(1, bomb.blast_strength):
                    if not explode(bomb.position.row, bomb.position.col - i, exploded_map, curr_board):
                        break
                for i in range(1, bomb.blast_strength):
                    if not explode(bomb.position.row, bomb.position.col + i, exploded_map, curr_board):
                        break

            for bomb in curr_bombs:
                if exploded_map[bomb.position.row][bomb.position.col] == 1:
                    bomb.fire()
                    has_new_explosions = True

        # Update the board's bombs.
        for bomb in curr_bombs:
            curr_board[bomb.position.row][bomb.position.col] = constants.Item.Bomb

        # Update the board's flames.
        for row in range(constants.BOARD_SIZE):
            for col in range(constants.BOARD_SIZE):
                if exploded_map[row][col] == 1:
                    curr_flames.append(characters.Flame.new(Position(row, col)))

        for flame in curr_flames:
            curr_board[flame.position.row][flame.position.col] = constants.Item.Flames

        # Kill agents on flames. Otherwise, update position on curr_board.
        for agent in alive_agents:
            if curr_board[agent.position.row][agent.position.col] == constants.Item.Flames:
                agent.die()
            else:
                curr_board[agent.position.row][agent.position.col] = utility.agent_value(agent.agent_id)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cdef np.ndarray get_features(self, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, list agents, list bombs,
                                 byte is_partially_observable, byte agent_view_size):
        # TODO: implement partial observability
        cdef np.ndarray features_np = np.zeros((4, constants.BOARD_SIZE, constants.BOARD_SIZE, 18), dtype=np.uint8)
        cdef unsigned char[:, :, :, :] features = features_np
        cdef byte agent_id, row, col, i
        cdef characters.Bomber agent
        cdef characters.Bomb bomb

        for agent_id in range(4):
            for row in range(constants.BOARD_SIZE):
                for col in range(constants.BOARD_SIZE):
                    for i in range(10):
                        features[agent_id, row, col, i] = (board[row][col] == i)

        for bomb in bombs:
            for agent_id in range(4):
                features[agent_id, bomb.position.row, bomb.position.col, 10] = bomb.blast_strength
                features[agent_id, bomb.position.row, bomb.position.col, 11] = bomb.life

        for agent in agents:
            for row in range(constants.BOARD_SIZE):
                for col in range(constants.BOARD_SIZE):
                    features[agent.agent_id, row, col, 12] = agent.ammo
                    features[agent.agent_id, row, col, 13] = agent.blast_strength
                    features[agent.agent_id, row, col, 14] = agent.can_kick
                    features[agent.agent_id, row, col, 16] = (board[row][col] == agent.teammate)
                    features[agent.agent_id, row, col, 17] = (board[row][col] in agent.enemies)
            features[agent.agent_id, agent.position.row, agent.position.col, 15] = 1

        return features_np

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cdef list get_observations(self, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] curr_board, list agents, list bombs,
                               byte is_partially_observable, byte agent_view_size):
        """Gets the observations as an np.array of the visible squares.

        The agent gets to choose whether it wants to keep the fogged part in
        memory.
        """
        cdef characters.Bomber agent
        cdef list alive_agents, observations
        cdef dict agent_obs
        cdef np.ndarray board_np
        cdef unsigned char[:, :] board

        alive_agents = [utility.agent_value(agent.agent_id)
                        for agent in agents if agent.is_alive]

        observations = []
        for agent in agents:
            agent_obs = {'alive': alive_agents}
            # create new numpy array and copy data there
            board_np = np.zeros((constants.BOARD_SIZE, constants.BOARD_SIZE), dtype=np.uint8)
            board = board_np
            memcpy(&board[0, 0], curr_board, board.nbytes)
            if is_partially_observable:
                for row in range(constants.BOARD_SIZE):
                    for col in range(constants.BOARD_SIZE):
                        if not in_view_range(agent.position, row, col, agent_view_size):
                            board[row][col] = constants.Item.Fog
            agent_obs['board'] = board_np
            bomb_blast_strengths, bomb_life = make_bomb_maps(agent.position, bombs, is_partially_observable, agent_view_size)
            agent_obs['bomb_blast_strength'] = bomb_blast_strengths
            agent_obs['bomb_life'] = bomb_life

            agent_obs['position'] = (agent.position.row, agent.position.col)
            agent_obs['blast_strength'] = agent.blast_strength
            agent_obs['can_kick'] = agent.can_kick
            agent_obs['teammate'] = agent.teammate
            agent_obs['ammo'] = agent.ammo
            agent_obs['enemies'] = agent.enemies
            observations.append(agent_obs)

        return observations

    cdef bint get_done(self, list agents, unsigned short step_count, unsigned short max_steps, constants.GameType game_type, byte training_agent):
        cdef characters.Bomber agent
        cdef list alive_ids
        cdef byte alive

        if step_count > max_steps:
            return True
        elif game_type == constants.GameType.FFA:
            alive = 0
            for agent in agents:
                if agent.is_alive:
                    alive += 1
                elif agent.agent_id == training_agent:
                    return True
            return alive <= 1
        else:
            # TODO: optimize non-FFA games
            alive_ids = sorted([agent.agent_id for agent in agents if agent.is_alive])
            return any([
                len(alive_ids) <= 1,
                alive_ids == [0, 2],
                alive_ids == [1, 3],
            ])

    cdef dict get_info(self, byte done, list rewards, constants.GameType game_type, list agents):
        cdef characters.Bomber agent

        if game_type == constants.GameType.FFA:
            alive = [agent for agent in agents if agent.is_alive]
            if done:
                if len(alive) != 1:
                    # Either we have more than 1 alive (reached max steps) or
                    # we have 0 alive (last agents died at the same time).
                    return {
                        'result': constants.Result.Tie,
                    }
                else:
                    return {
                        'result': constants.Result.Win,
                        'winners': [num for num, reward in enumerate(rewards)
                                    if reward == 1]
                    }
            else:
                return {
                    'result': constants.Result.Incomplete,
                }
        elif done:
            # We are playing a team game.
            if rewards == [-1] * 4:
                return {
                    'result': constants.Result.Tie,
                }
            else:
                return {
                    'result': constants.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards)
                                if reward == 1],
                }
        else:
            return {
                'result': constants.Result.Incomplete,
            }

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cdef np.ndarray get_rewards(self, list agents, constants.GameType game_type, unsigned short step_count, unsigned short max_steps):
        cdef characters.Bomber agent
        cdef list alive_agents
        cdef byte alive
        cdef rewards_np = np.zeros(4, dtype=np.float32)
        cdef float[:] rewards = rewards_np

        if game_type == constants.GameType.FFA:
            alive = 0
            for agent in agents:
                rewards[agent.agent_id] = agent.is_alive
                if agent.is_alive:
                    alive += 1
            if alive == 1:
                # An agent won. Give them +1, others -1.
                for i in range(4):
                    rewards[i] = rewards[i] * 2 - 1
            elif step_count > max_steps:
                # Game is over from time. Everyone gets -1.
                rewards[:] = -1
            else:
                # Game running: 0 for alive, -1 for dead.
                for i in range(4):
                    rewards[i] = rewards[i] - 1
            return rewards_np
        else:
            # TODO: optimize non-FFA games
            alive_agents = [num for num, agent in enumerate(agents)
                            if agent.is_alive]
            # We are playing a team game.
            if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
                # Team [0, 2] wins.
                rewards_np[:] = [1, -1, 1, -1]
                return rewards_np
            elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
                # Team [1, 3] wins.
                rewards_np[:] = [-1, 1, -1, 1]
                return rewards_np
            elif step_count >= max_steps:
                # Game is over by max_steps. All agents tie.
                rewards_np[:] = [-1, -1, -1, -1]
                return rewards_np
            else:
                # No team has yet won or lost.
                rewards_np[:] = [0, 0, 0, 0]
                return rewards_np
