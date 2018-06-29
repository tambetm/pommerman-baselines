"""The set of constants in the game.

This includes not just ints but also classes like Item, GameType, Action, etc.
"""

cdef enum:
    RENDER_FPS = 15
    BOARD_SIZE = 11
    NUM_RIGID = 36
    NUM_WOOD = 36
    NUM_ITEMS = 20
    AGENT_VIEW_SIZE = 4
    HUMAN_FACTOR = 32
    DEFAULT_BLAST_STRENGTH = 2
    DEFAULT_BOMB_LIFE = 10

    # If using collapsing boards, the step at which the board starts to collapse.
    FIRST_COLLAPSE = 500
    MAX_STEPS = 800
    RADIO_VOCAB_SIZE = 8
    RADIO_NUM_WORDS = 2

    MAXBUF = 1024


cdef enum Item:
    Passage = 0
    Rigid = 1
    Wood = 2
    Bomb = 3
    Flames = 4
    Fog = 5
    ExtraBomb = 6
    IncrRange = 7
    Kick = 8
    AgentDummy = 9
    Agent0 = 10
    Agent1 = 11
    Agent2 = 12
    Agent3 = 13
    NumItems = 14


cdef enum GameType:
    FFA = 1
    Team = 2
    TeamRadio = 3


cdef enum Action:
    Stop = 0
    Up = 1
    Down = 2
    Left = 3
    Right = 4
    LayBomb = 5


cdef enum Result:
    Win = 0
    Loss = 1
    Tie = 2
    Incomplete = 3


cdef class InvalidAction(Exception):
    pass
