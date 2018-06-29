cimport constants
cimport numpy as np

from utility cimport byte

cdef class ForwardModel(object):

    #cdef list act(self, list agents, list obs, object action_space, byte is_communicative=*)
    cdef void step(self, const unsigned char[:] actions,
                   byte[constants.BOARD_SIZE][constants.BOARD_SIZE] curr_board,
                   byte[constants.BOARD_SIZE][constants.BOARD_SIZE] curr_items,
                   list curr_agents,
                   list curr_bombs,
                   list curr_flames,
                   byte max_blast_strength=*)
    cdef np.ndarray get_features(self, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, list agents, list bombs,
                                 byte is_partially_observable, byte agent_view_size)
    cdef list get_observations(self, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] curr_board, list agents, list bombs,
                               byte is_partially_observable, byte agent_view_size)
    cdef bint get_done(self, list agents, unsigned short step_count, unsigned short max_steps, constants.GameType game_type, byte training_agent)
    cdef dict get_info(self, byte done, list rewards, constants.GameType game_type, list agents)
    cdef np.ndarray get_rewards(self, list agents, constants.GameType game_type, unsigned short step_count, unsigned short max_steps)
