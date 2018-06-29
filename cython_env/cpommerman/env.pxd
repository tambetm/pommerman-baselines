cimport constants
cimport forward_model
cimport numpy as np

from utility cimport byte

cdef class Pomme(object):
    cdef byte _render_fps
    cdef constants.GameType _game_type
    cdef byte _board_size
    cdef byte _agent_view_size
    cdef byte _num_rigid
    cdef byte _num_wood
    cdef byte _num_items
    cdef unsigned short _step_count
    cdef unsigned short _max_steps
    cdef byte _is_partially_observable
    cdef byte training_agent
    cdef list _agents, _bombs, _flames
    cdef forward_model.ForwardModel model
    cdef object action_space, observation_space
    cdef byte[constants.BOARD_SIZE][constants.BOARD_SIZE] _board
    cdef byte[constants.BOARD_SIZE][constants.BOARD_SIZE] _items

    cdef void _set_action_space(self)
    cdef void _set_observation_space(self)
    cdef void set_agents(self, list agents)
    cdef void make_board(self)
    cdef void make_items(self)
    #cdef list act(self, list obs)
    cpdef np.ndarray get_features(self)
    cpdef list get_observations(self)
    cpdef np.ndarray get_rewards(self)
    cpdef bint get_done(self)
    cpdef dict get_info(self, byte done, list rewards)
    cpdef void reset(self)
    #cdef list seed(self, seed=*)
    cpdef void step(self, const unsigned char[:] actions)
    cpdef dict get_json_info(self)
    cpdef void set_json_info(self, dict game_state)
    cpdef bytes get_state(self)
    cpdef void set_state(self, bytes state)
