cimport constants
cimport numpy as np

ctypedef signed char byte

cdef struct Position:
    byte row
    byte col

cdef byte Position_eq(Position self, Position other)
cdef byte Position_neq(Position self, Position other)

cdef void make_board(byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, byte num_rigid, byte num_wood)
cdef void make_items(byte[constants.BOARD_SIZE][constants.BOARD_SIZE] items, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, byte num_items)
cdef byte is_valid_direction(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position, constants.Action direction)
cdef byte _position_is_item(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position, constants.Item item)
cdef byte position_is_flames(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position)
cdef byte position_is_bomb(list bombs, Position position)
cdef byte position_is_powerup(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position)
cdef byte position_is_wall(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position)
cdef constants.Item agent_value(byte id_)
cdef byte position_on_board(const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] board, Position position)
cdef Position get_next_position(Position position, constants.Action direction)

cdef void* encode(void *dest, const void* src, int length)
cdef const void* decode(const void *src, void* dest, int length)
