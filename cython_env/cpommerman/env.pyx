"""The baseline Pommerman environment.

This evironment acts as game manager for Pommerman. Further environments,
such as in v1.py, will inherit from this.
"""
import json
import os

import numpy as np
cimport numpy as np
import time
from gym import spaces
from gym.utils import seeding
import gym

cimport cython
cimport characters
cimport constants
cimport forward_model
import utility
cimport utility
from utility cimport Position, Position, byte

from libc.string cimport memset
from libc.string cimport memcpy


cdef class Pomme(object):

    def __init__(self,
                 render_fps=None,
                 game_type=None,
                 board_size=None,
                 agent_view_size=10,
                 num_rigid=None,
                 num_wood=None,
                 num_items=None,
                 max_steps=1000,
                 is_partially_observable=False,
                 **kwargs):
        self._render_fps = render_fps
        self._agents = None
        self._game_type = game_type
        self._board_size = board_size
        self._agent_view_size = agent_view_size
        self._num_rigid = num_rigid
        self._num_wood = num_wood
        self._num_items = num_items
        self._max_steps = max_steps
        self._is_partially_observable = is_partially_observable

        self.training_agent = -1
        self.model = forward_model.ForwardModel()

        # Observation and Action Spaces. These are both geared towards a single
        # agent even though the environment expects actions and returns
        # observations for all four agents. We do this so that it's clear what
        # the actions and obs are for a single agent. Wrt the observations,
        # they are actually returned as a dict for easier understanding.
        self._set_action_space()
        self._set_observation_space()

    cdef void _set_action_space(self):
        self.action_space = spaces.Discrete(6)

    cdef void _set_observation_space(self):
        """The Observation Space for each agent.

        There are a total of 3*board_size^2+12 observations:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy, Agent3}).
        - enemies (three of {AgentDummy, Agent3}).
        """
        cdef int bss
        cdef list min_obs, max_obs

        bss = self._board_size**2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy] * 4
        max_obs = [constants.Item.NumItems] * bss + [self._board_size] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3] * 4
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    cdef void set_agents(self, list agents):
        self._agents = agents

    cdef void make_board(self):
        utility.make_board(self._board, self._num_rigid, self._num_wood)

    cdef void make_items(self):
        utility.make_items(self._items, self._board, self._num_items)
    '''
    cdef list act(self, list obs):
        cdef list agents
        cdef characters.Bomber agent
        agents = [agent for agent in self._agents \
                  if agent.agent_id != self.training_agent]
        return forward_model.ForwardModel.act(agents, obs, self.action_space)
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef np.ndarray get_features(self):
        features = self.model.get_features(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size or 10)
        return features

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef list get_observations(self):
        observations = self.model.get_observations(
            self._board, self._agents, self._bombs,
            self._is_partially_observable, self._agent_view_size or 10)
        return observations

    cpdef np.ndarray get_rewards(self):
        return self.model.get_rewards(self._agents, self._game_type,
                                      self._step_count, self._max_steps)

    cpdef bint get_done(self):
        return self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agent)

    cpdef dict get_info(self, byte done, list rewards):
        return self.model.get_info(done, rewards, self._game_type, self._agents)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef void reset(self):
        cdef byte agent_id, row = 0, col = 0
        cdef characters.Bomber agent
        cdef byte stop

        assert (self._agents is not None)

        self._step_count = 0
        memset(self._board, 0, sizeof(self._board))
        self.make_board()
        memset(self._items, 0, sizeof(self._items))
        self.make_items()
        self._bombs = []
        self._flames = []
        for agent_id, agent in enumerate(self._agents):
            stop = False
            for row in range(constants.BOARD_SIZE):
                for col in range(constants.BOARD_SIZE):
                    if self._board[row][col] == utility.agent_value(agent_id):
                        stop = True
                        break
                if stop:
                    break
            else:
                assert False
            agent.set_start_position(Position(row, col))
            agent.reset()
    '''
    cdef list seed(self, seed=None):
        gym.spaces.prng.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    '''
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef void step(self, const unsigned char[:] actions):
        cdef byte max_blast_strength
        cdef byte done
        cdef list obs, reward
        cdef dict info

        max_blast_strength = self._agent_view_size or 10
        self.model.step(
            actions,
            self._board,
            self._items,
            self._agents,
            self._bombs,
            self._flames,
            max_blast_strength=max_blast_strength)

        self._step_count += 1

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef dict get_json_info(self):
        """Returns a json snapshot of the current game state."""
        cdef dict ret
        cdef byte r, c
        cdef str key
        cdef object value

        ret = {
            'board_size': self._board_size,
            'step_count': self._step_count,
            'board': [[self._board[r][c] for c in range(constants.BOARD_SIZE)] for r in range(constants.BOARD_SIZE)],
            'agents': self._agents,
            'bombs': self._bombs,
            'flames': self._flames,
            'items': [[(r, c), self._items[r][c]] for c in range(constants.BOARD_SIZE) for r in range(constants.BOARD_SIZE) if self._items[r][c] != 0]
        }
        for key, value in ret.items():
            ret[key] = json.dumps(value, cls=utility.PommermanJSONEncoder)
        return ret

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef void set_json_info(self, dict game_state):
        cdef byte x, y, board_size
        cdef characters.Bomber agent
        cdef constants.Action moving_direction
        cdef list i, board_array, item_array, agent_array, bomb_array, flame_array
        cdef dict a, b, f

        """Sets the game state as the game_state."""
        board_size = int(game_state['board_size'])
        self._board_size = board_size
        self._step_count = int(game_state['step_count'])

        board_array = json.loads(game_state['board'])
        for x in range(board_size):
            for y in range(board_size):
                self._board[x][y] = board_array[x][y]

        item_array = json.loads(game_state['items'])
        memset(self._items, 0, sizeof(self._items))
        for i in item_array:
            x, y = i[0]
            self._items[x][y] = i[1]

        agent_array = json.loads(game_state['agents'])
        for a in agent_array:
            for agent in self._agents:
                if agent.agent_id == a['agent_id']:
                    break
            else:
                assert False
            agent.set_start_position(Position(a['position'][0], a['position'][1]))
            agent.reset(
                int(a['ammo']), bool(a['is_alive']), int(a['blast_strength']),
                bool(a['can_kick']))

        self._bombs = []
        bomb_array = json.loads(game_state['bombs'])
        for b in bomb_array:
            for agent in self._agents:
                if agent.agent_id == b['bomber_id']:
                    break
            else:
                assert False
            if b['moving_direction'] is None:
                moving_direction = constants.Action.Stop
            else:
                moving_direction = int(b['moving_direction'])
            self._bombs.append(characters.Bomb.new(
                agent, Position(b['position'][0], b['position'][1]), int(b['life']),
                int(b['blast_strength']), moving_direction)
            )

        self._flames = []
        flame_array = json.loads(game_state['flames'])
        for f in flame_array:
            self._flames.append(
                characters.Flame.new(Position(f['position'][0], f['position'][1]), f['life']))

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef bytes get_state(self):
        cdef bytes _buf = bytes(constants.MAXBUF)
        cdef characters.Bomber agent
        cdef characters.Bomb bomb
        cdef characters.Flame flame
        cdef void *buf = <byte*>_buf
        cdef void *init_buf = buf
        cdef byte num

        buf = utility.encode(buf, &self._step_count, sizeof(self._step_count))
        buf = utility.encode(buf, self._board, sizeof(self._board))
        buf = utility.encode(buf, self._items, sizeof(self._items))

        num = len(self._agents)
        buf = utility.encode(buf, &num, sizeof(num))
        for agent in self._agents:
            buf = agent.encode(buf)

        num = len(self._bombs)
        buf = utility.encode(buf, &num, sizeof(num))
        for bomb in self._bombs:
            buf = bomb.encode(buf)

        num = len(self._flames)
        buf = utility.encode(buf, &num, sizeof(num))
        for flame in self._flames:
            buf = flame.encode(buf)

        assert buf - init_buf <= constants.MAXBUF
        return _buf[:(buf - init_buf)]

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.initializedcheck(False)   # Deactivate memoryview initialization check.
    cpdef void set_state(self, bytes state):
        cdef const void *buf = <const byte*>state
        cdef const void *init_buf = buf
        cdef characters.Bomber agent
        cdef characters.Bomb bomb
        cdef characters.Flame flame
        cdef byte num, agent_id

        buf = utility.decode(buf, &self._step_count, sizeof(self._step_count))
        buf = utility.decode(buf, self._board, sizeof(self._board))
        buf = utility.decode(buf, self._items, sizeof(self._items))

        buf = utility.decode(buf, &num, sizeof(num))
        for i in range(num):
            agent = self._agents[i]
            buf = agent.decode(buf)

        self._bombs = []
        buf = utility.decode(buf, &num, sizeof(num))
        for i in range(num):
            bomb = characters.Bomb.__new__(characters.Bomb)
            # decode bomb owner here
            buf = utility.decode(buf, &agent_id, sizeof(agent_id))
            bomb.bomber = self._agents[agent_id]
            buf = bomb.decode(buf)
            self._bombs.append(bomb)

        self._flames = []
        buf = utility.decode(buf, &num, sizeof(num))
        for i in range(num):
            flame = characters.Flame.__new__(characters.Flame)
            buf = flame.decode(buf)
            self._flames.append(flame)
