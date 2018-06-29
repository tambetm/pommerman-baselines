cimport constants
cimport characters
from env cimport Pomme


cpdef make():
    cdef Pomme env
    cdef list agents

    env = Pomme(
        game_type=constants.GameType.FFA,
        board_size=constants.BOARD_SIZE,
        num_rigid=constants.NUM_RIGID,
        num_wood=constants.NUM_WOOD,
        num_items=constants.NUM_ITEMS,
        max_steps=constants.MAX_STEPS,
        render_fps=constants.RENDER_FPS,
    )

    agents = [characters.Bomber.new(id, constants.GameType.FFA) for id in range(4)]
    env.set_agents(agents)
    return env
