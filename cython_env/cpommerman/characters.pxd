cimport constants
from utility cimport Position, byte

cdef class Bomber(object):
    cdef constants.GameType _game_type
    cdef byte ammo
    cdef byte is_alive
    cdef byte blast_strength
    cdef byte can_kick
    cdef byte agent_id
    cdef constants.Item teammate
    cdef constants.Item[3] enemies
    cdef Position position
    cdef Position start_position
    cdef Position desired_position
    cdef Position delayed_position
    cdef Bomb kicked_bomb

    @staticmethod
    cdef Bomber new(byte agent_id=*, constants.GameType game_type=*)
    cdef void set_agent_id(self, byte agent_id)
    cdef Bomb maybe_lay_bomb(self)
    cdef void incr_ammo(self)
    cdef Position get_next_position(self, constants.Action direction)
    cdef void move(self, constants.Action direction)
    cdef void stop(self)
    cdef byte in_range(self, const byte[constants.BOARD_SIZE][constants.BOARD_SIZE] exploded_map)
    cdef void die(self)
    cdef void set_start_position(self, Position start_position)
    cdef void reset(self, byte ammo=*, byte is_alive=*, byte blast_strength=*, byte can_kick=*)
    cdef void pick_up(self, constants.Item item, byte max_blast_strength)
    cdef dict to_json(self)
    cdef void *encode(self, void *buf)
    cdef const void *decode(self, const void *buf)


cdef class Bomb(object):
    cdef Bomber bomber
    cdef Position position
    cdef byte life
    cdef byte blast_strength
    cdef constants.Action moving_direction
    cdef Position desired_position
    cdef Position delayed_position
    cdef Bomber kicked_agent

    @staticmethod
    cdef Bomb new(Bomber bomber,
                  Position position,
                  byte life,
                  byte blast_strength,
                  constants.Action moving_direction=*)
    cdef void tick(Bomb self)
    cdef void fire(Bomb self)
    cdef void move(Bomb self)
    cdef void stop(Bomb self)
    cdef byte exploded(Bomb self)
    cdef byte is_moving(Bomb self)
    cdef dict to_json(Bomb self)
    cdef void *encode(self, void *buf)
    cdef const void *decode(self, const void *buf)

cdef class Flame(object):
    cdef Position position
    cdef byte _life

    @staticmethod
    cdef Flame new(Position position, byte life=*)
    cdef void tick(Flame self)
    cdef byte is_dead(Flame self)
    cdef dict to_json(Flame self)
    cdef void *encode(self, void *buf)
    cdef const void *decode(self, const void *buf)
