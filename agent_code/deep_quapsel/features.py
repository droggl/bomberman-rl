import numpy as np
from queue import Queue

import settings as s


def bomb_field(field, bombs):
    """
    Draw a map of spots threatened by bombs.
    no bomb = 10, else value = bomb countdown

    :param field: Board
    :param bombs: Bomb positions
    """
    (width, heigth) = np.shape(field)

    bomb_map = np.full((width, heigth), 10)

    for ((x,y), countdown) in bombs:
        bomb_map[x,y] = countdown
        i = 1
        while i <= s.BOMB_POWER and field[x+i, y] != -1:
            bomb_map[x+i,y] = min(countdown, bomb_map[x+i, y])
            i += 1
        i = 1
        while i <= s.BOMB_POWER and field[x, y+i] != -1:
            bomb_map[x,y+i] = min(countdown, bomb_map[x, y+i])
            i += 1
        i = 1
        while i <= s.BOMB_POWER and field[x-i, y] != -1:
            bomb_map[x-i,y] = min(countdown, bomb_map[x-i, y])
            i += 1
        i = 1
        while i <= s.BOMB_POWER and field[x, y-i] != -1:
            bomb_map[x,y-i] = min(countdown, bomb_map[x, y-i])
            i += 1

    return bomb_map


def danger_rating(field, bomb_map, explosion_map, start_pos, deadly):
    """
    Find shortest path from start pos to 'safe' (not threatened by bomb) field and returns distance.
    Does not use other player postions.
    Returns deadly if field itself is or not safe field can be reached.

    Used for survival_instinst().
    """
    # default high value if start_pos is deadly
    if bomb_map[start_pos] == 0 or explosion_map[start_pos] > 0:
        return deadly
    # else find shortest path to safety
    else:
        # setup parent matrix
        (width, heigth) = np.shape(field)
        parents = np.full((width, heigth), -1)
        # setup BFS FIFO
        q = Queue()
        q.put((start_pos, 0))

        shortest_dist = deadly

        while not q.empty():
            (pos, dist) = q.get()
            # limit search depth
            if dist > s.BOMB_TIMER:
                shortest_dist = 0
                break
            # already visited
            if parents[pos] > -1:
                continue
            parents[pos] = 0
            # found safe field
            if bomb_map[pos] == 10:
                shortest_dist = dist
                break

            viable = lambda x, y: (
                field[x, y] == 0 and        # field does not contain wall or crate
                (bomb_map[x, y] > dist + 1 or   # not exploded yet
                bomb_map[x, y] < dist   # already faded
                )
            )

            # traverse to viable fields
            (x,y) = pos
            if viable(x+1, y):
                q.put(((x+1, y), dist + 1))
            if viable(x, y+1):
                q.put(((x, y+1), dist + 1))
            if viable(x-1, y):
                q.put(((x-1, y), dist + 1))
            if viable(x, y-1):
                q.put(((x, y-1), dist + 1))

        return shortest_dist


def survival_instinct(field, bombs, explosion_map, others, player_pos):
    """
    Feature designed to help avoiding bombs.

    Fields are assigned a danger value, derived from distance to a safe field.
    First 4 values correspond to neighbours, last to field itself.

    :param field: Board
    :param bombs: Bomb positions
    :param explosion_map: Explosion_positions
    :param player_pos: Player position
    """
    danger = np.full(6, -1)

    # danger value for instadeath
    # TODO: find good value
    deadly = s.BOMB_POWER + 2

    (x,y) = player_pos

    bomb_map = bomb_field(field, bombs)

    other_pos = []
    for player in others:
        other_pos.append(player[3])

    viable = lambda x, y: (
        field[x,y] == 0 and         # field does not contain wall or crate
        not (x,y) in other_pos      # field not blocked by other player
    )

    # bfs to find safety for each neighbour
    if len(bombs) > 0:
        if viable(x+1, y):
            danger[0] = danger_rating(field, bomb_map, explosion_map, (x+1, y), deadly)
        if viable(x, y+1):
            danger[1] = danger_rating(field, bomb_map, explosion_map, (x, y+1), deadly)
        if viable(x-1, y):
            danger[2] = danger_rating(field, bomb_map, explosion_map, (x-1, y), deadly)
        if viable(x, y-1):
            danger[3] = danger_rating(field, bomb_map, explosion_map, (x, y-1), deadly)

        # waited
        danger[4] = danger_rating(field, bomb_map, explosion_map, player_pos, deadly)
    else:
        danger[4] = 0

    # dropped bomb
    bomb_map_drop = bomb_field(field, bombs + [(player_pos, 4)])
    danger[5] = danger_rating(field, bomb_map_drop, explosion_map, player_pos, deadly)

    # not viable moves have same danger as staying
    danger[danger == -1] = danger[4]

    return danger / deadly


def crate_and_enemy_potential(field, bomb_pos, others):
    """
    Counts how many crates and enemies a bomb dropped at player_pos would destroy.

    :param field: Board.
    :param player_pos: Player coordinates.
    """
    crates = 0
    enemies = 0

    other_pos = []
    for player in others:
        other_pos.append(player[3])

    (x,y) = bomb_pos

    # go in each direction and count crates
    bomb_range = 1+s.BOMB_POWER
    bomb_range = 2

    x_neg = lambda x, y, i: (x-i, y)
    x_pos = lambda x, y, i: (x+i, y)
    y_neg = lambda x, y, i: (x, y-i)
    y_pos = lambda x, y, i: (x, y+i)

    for get_xy in (x_neg, x_pos, y_neg, y_pos):
        for i in range(1, bomb_range):
            cur_x, cur_y = get_xy(x,y,i)
            if field[cur_x, cur_y] == -1:
                break
            if field[cur_x, cur_y] == 1:
                crates += 1
            elif (cur_x, cur_y) in others :
                enemies += 1

    return crates, enemies


def object_distance(field, objects, pos):
    """
    Distance to nearest reachable object in objects list. 
    0 if none reachable.
    Pathfinding does ignore bombs and other players.

    :param field: Board
    :param coins: Objects positions
    :param player_pos: Start position
    """
    if len(objects) == 0:
        return 0

    distance = 0

    # setup parent matrix
    dim = np.shape(field)
    parents = np.full(dim, -1)

    # setup BFS fifo
    q = Queue()
    q.put((pos, 1))

    while not q.empty():
        (pos, dist) = q.get()
        # max search depth
        # if dist > max:
        #     break

        # already visited
        if parents[pos] > -1:
            continue
        parents[pos] = 0

        # found coin
        if pos in objects:
            distance = dist
            break

        # traverse
        (x,y) = pos
        if field[x+1, y] == 0:
            q.put(((x+1,y), dist+1))
        if field[x, y+1] == 0:
            q.put(((x,y+1), dist+1))
        if field[x-1, y] == 0:
            q.put(((x-1,y), dist+1))
        if field[x, y-1] == 0:
            q.put(((x,y-1), dist+1))

    return distance


def crate_distance(field, start_pos):
    """
    Distance to closest crate.

    :param field: Board
    :param start_pos: Start pos for search
    """
    distance = 0

    # setup parent matrix
    dim = np.shape(field)
    parents = np.full(dim, -1)

    # setup BFS fifo
    q = Queue()
    q.put((start_pos, 1))

    while not q.empty():
        (pos, dist) = q.get()
        # max search depth
        # if dist > max:
        #     break

        # already visited
        if parents[pos] > -1:
            continue
        parents[pos] = 0

        # found coin
        if field[pos] == 1:
            distance = dist
            break

        # traverse
        (x,y) = pos
        if field[x+1, y] != -1:
            q.put(((x+1,y), dist+1))
        if field[x, y+1] != -1:
            q.put(((x,y+1), dist+1))
        if field[x-1, y] != -1:
            q.put(((x-1,y), dist+1))
        if field[x, y-1] != -1:
            q.put(((x,y-1), dist+1))

    return distance
