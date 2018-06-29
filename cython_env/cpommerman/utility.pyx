import itertools
import json
import random

from gym import spaces
import numpy as np
cimport numpy as np
cimport cython
cimport constants
cimport characters

from libc.string cimport memcpy


class PommermanJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        cdef characters.Bomber agent
        cdef characters.Bomb bomb
        cdef characters.Flame flame
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, characters.Bomber):
            agent = <characters.Bomber>obj
            return agent.to_json()
        if isinstance(obj, characters.Bomb):
            bomb = <characters.Bomb>obj
            return bomb.to_json()
        if isinstance(obj, characters.Flame):
            flame = <characters.Flame>obj
            return flame.to_json()
        #elif isinstance(obj, constants.Item):
        #    return obj
        #elif isinstance(obj, constants.Action):
        #    return obj
        elif isinstance(obj, np.uint8):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif isinstance(obj, spaces.Discrete):
            return obj.n
        elif isinstance(obj, spaces.Tuple):
            return [space.n for space in obj.spaces]
        return json.JSONEncoder.default(self, obj)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef byte lay_wall(constants.Item value, byte num_left, set coordinates, byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board):
    cdef byte x, y
    # TODO: sample using C?
    x, y = random.sample(coordinates, 1)[0]
    coordinates.remove((x, y))
    coordinates.remove((y, x))
    board[x][y] = value
    board[y][x] = value
    num_left -= 2
    return num_left


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef list make(byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, byte num_rigid, byte num_wood):
    cdef set coordinates
    cdef byte x, y, i
    cdef list agents
    cdef byte size = constants.BOARD_SIZE
    # Initialize everything as a passage.

    # Gather all the possible coordinates to use for walls.
    coordinates = set()
    for x in range(size):
        for y in range(size):
            if x != y:
                coordinates.add((x, y))

    # Set the players down. Exclude them from coordinates.
    # Agent0 is in top left. Agent1 is in bottom left.
    # Agent2 is in bottom right. Agent 3 is in top right.
    board[1][1] = constants.Item.Agent0
    board[size - 2][1] = constants.Item.Agent1
    board[size - 2][size - 2] = constants.Item.Agent2
    board[1][size - 2] = constants.Item.Agent3
    agents = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
    for position in agents:
        if position in coordinates:
            coordinates.remove(position)

    # Exclude breathing room on either side of the agents.
    for i in range(2, 4):
        coordinates.remove((1, i))
        coordinates.remove((i, 1))
        coordinates.remove((1, size - i - 1))
        coordinates.remove((size - i - 1, 1))
        coordinates.remove((size - 2, size - i - 1))
        coordinates.remove((size - i - 1, size - 2))
        coordinates.remove((i, size - 2))
        coordinates.remove((size - 2, i))

    # Lay down wooden walls providing guaranteed passage to other agents.
    for i in range(4, size - 4):
        board[1][i] = constants.Item.Wood
        board[size - i - 1][1] = constants.Item.Wood
        board[size - 2][size - i - 1] = constants.Item.Wood
        board[size - i - 1][size - 2] = constants.Item.Wood
        coordinates.remove((1, i))
        coordinates.remove((size - i - 1, 1))
        coordinates.remove((size - 2, size - i - 1))
        coordinates.remove((size - i - 1, size - 2))
        num_wood -= 4

    # Lay down the rigid walls.
    while num_rigid > 0:
        num_rigid = lay_wall(constants.Item.Rigid, num_rigid,
                             coordinates, board)

    # Lay down the wooden walls.
    while num_wood > 0:
        num_wood = lay_wall(constants.Item.Wood, num_wood,
                            coordinates, board)

    return agents


cdef void make_board(byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, byte num_rigid, byte num_wood):
    """Make the random but symmetric board.

    The numbers refer to the Item enum in constants. This is:
     0 - passage
     1 - rigid wall
     2 - wood wall
     3 - bomb
     4 - flames
     5 - fog
     6 - extra bomb item
     7 - extra firepower item
     8 - kick
     9 - skull
     10 - 13: agents

    Args:
      size: The dimension of the board, i.e. it's sizeXsize.
      num_rigid: The number of rigid walls on the board. This should be even.
      num_wood: Similar to above but for wood walls.

    Returns:
      board: The resulting random board.
    """

    assert (num_rigid % 2 == 0)
    assert (num_wood % 2 == 0)
    agents = make(board, num_rigid, num_wood)

    # Make sure it's possible to reach most of the passages.
    while len(inaccessible_passages(board, agents)) > 4:
        agents = make(board, num_rigid, num_wood)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef void make_items(byte[constants.BOARD_SIZE][constants.BOARD_SIZE] items, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, byte num_items):
    cdef byte row, col
    while num_items > 0:
        # TODO: use C for random?
        row = random.randint(0, constants.BOARD_SIZE - 1)
        col = random.randint(0, constants.BOARD_SIZE - 1)
        if board[row][col] != constants.Item.Wood:
            continue
        if items[row][col] != 0:
            continue

        # TODO: use C for random?
        items[row][col] = random.choice([
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ])
        num_items -= 1


# TODO: rewrite this? maybe not crucial, because used only once per episode
cdef list inaccessible_passages(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, list agent_positions):
    """Return inaccessible passages on this board."""
    cdef set seen
    cdef tuple agent_position, next_position
    cdef Position next_position_c
    cdef list positions, Q
    cdef byte row, col, i, j

    seen = set()
    agent_position = agent_positions.pop()
    positions = []
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board[i][j] == constants.Item.Passage:
                positions.append((i, j))

    Q = [agent_position]
    while Q:
        row, col = Q.pop()
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_position = (row + i, col + j)
            next_position_c = Position(row + i, col + j)
            if next_position in seen:
                continue
            if not position_on_board(board, next_position_c):
                continue
            if position_is_rigid(board, next_position_c):
                continue

            if next_position in positions:
                positions.pop(positions.index(next_position))
                if not len(positions):
                    return []

            seen.add(next_position)
            Q.append(next_position)
    return positions


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef byte is_valid_direction(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position, constants.Action direction):
    if direction == constants.Action.Stop:
        return True

    if direction == constants.Action.Up:
        return position.row - 1 >= 0 and board[position.row - 1][position.col] not in (constants.Item.Rigid, constants.Item.Wood)

    if direction == constants.Action.Down:
        return position.row + 1 < constants.BOARD_SIZE and board[position.row + 1][position.col] not in (constants.Item.Rigid, constants.Item.Wood)

    if direction == constants.Action.Left:
        return position.col - 1 >= 0 and board[position.row][position.col - 1] not in (constants.Item.Rigid, constants.Item.Wood)

    if direction == constants.Action.Right:
        return position.col + 1 < constants.BOARD_SIZE and \
            board[position.row][position.col + 1] not in (constants.Item.Rigid, constants.Item.Wood)

    raise constants.InvalidAction("We did not receive a valid direction: ",
                                  direction)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef inline byte _position_is_item(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position, constants.Item item):
    return board[position.row][position.col] == item


cdef inline byte position_is_flames(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return _position_is_item(board, position, constants.Item.Flames)


cdef byte position_is_bomb(list bombs, Position position):
    """Check if a given position is a bomb.
    
    We don't check the board because that is an unreliable source. An agent
    may be obscuring the bomb on the board.
    """
    cdef characters.Bomb bomb
    # TODO: get rid of list?
    for bomb in bombs:
        if Position_eq(position, bomb.position):
            return True
    return False


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   # Deactivate memoryview initialization check.
cdef byte position_is_powerup(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return board[position.row][position.col] in (constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick)


cdef byte position_is_wall(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return position_is_rigid(board, position) or \
        position_is_wood(board, position)


cdef inline byte position_is_passage(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return _position_is_item(board, position, constants.Item.Passage)


cdef byte position_is_rigid(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return _position_is_item(board, position, constants.Item.Rigid)


cdef byte position_is_wood(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return _position_is_item(board, position, constants.Item.Wood)

'''
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef byte position_is_agent(const unsigned char[:, :] board, tuple position):
    cdef int x, y
    x, y = position
    return board[x, y] in [
        constants.Item.Agent0, constants.Item.Agent1,
        constants.Item.Agent2, constants.Item.Agent3
    ]
'''
'''
cdef byte position_is_enemy(const unsigned char[:, :] board, tuple position, list enemies):
    cdef int x, y
    x, y = position
    return board[x, y] in enemies
'''
'''
# TODO: Fix this so that it includes the teammate.
cdef byte position_is_passable(const unsigned char[:, :] board, tuple position, list enemies):
    return (position_is_agent(board, position) or
            position_is_powerup(board, position) or
            position_is_passage(board, position))\
        and not position_is_enemy(board, position, enemies)
'''

cdef inline byte position_is_fog(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return _position_is_item(board, position, constants.Item.Fog)


cdef inline constants.Item agent_value(byte id_):
    return <constants.Item>(id_ + constants.Item.Agent0)


#cdef byte position_in_items(const unsigned char[:, :] board, tuple position, list items):
#    return any([_position_is_item(board, position, item) for item in items])


cdef inline byte position_on_board(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position):
    return constants.BOARD_SIZE > position.row and constants.BOARD_SIZE > position.col and position.row >= 0 and position.col >= 0

'''
cdef constants.Action get_direction(tuple position, tuple next_position):
    """Get the direction such that position --> next_position.

    We assume that they are adjacent.
    """
    cdef byte x, y, nx, ny
    x, y = position
    nx, ny = next_position
    if x == nx:
        if y < ny:
            return constants.Action.Right
        else:
            return constants.Action.Left
    elif y == ny:
        if x < nx:
            return constants.Action.Down
        else:
            return constants.Action.Up
    raise constants.InvalidAction(
        "We did not receive a valid position transition.")
'''

cdef Position get_next_position(Position position, constants.Action direction):
    if direction == constants.Action.Right:
        return Position(position.row, position.col + 1)
    elif direction == constants.Action.Left:
        return Position(position.row, position.col - 1)
    elif direction == constants.Action.Down:
        return Position(position.row + 1, position.col)
    elif direction == constants.Action.Up:
        return Position(position.row - 1, position.col)
    elif direction == constants.Action.Stop:
        return Position(position.row, position.col)
    raise constants.InvalidAction("We did not receive a valid direction.")

cdef byte Position_eq(Position self, Position other):
    return (self.row == other.row and self.col == other.col)

cdef byte Position_neq(Position self, Position other):
    return (self.row != other.row or self.col != other.col)

cdef inline void* encode(void *dest, const void* src, int length):
    memcpy(dest, src, length)
    return dest + length

cdef inline const void* decode(const void *src, void* dest, int length):
    memcpy(dest, src, length)
    return src + length
