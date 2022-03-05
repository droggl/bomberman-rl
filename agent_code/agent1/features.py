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
            i = i + 1
        i = 1
        while i <= s.BOMB_POWER and field[x, y+i] != -1:
            bomb_map[x,y+i] = min(countdown, bomb_map[x, y+i])
            i = i+1
        i = 1
        while i <= s.BOMB_POWER and field[x-i, y] != -1:
            bomb_map[x-i,y] = min(countdown, bomb_map[x-i, y])
            i = i+1
        i = 1
        while i <= s.BOMB_POWER and field[x, y-i] != -1:
            bomb_map[x,y-i] = min(countdown, bomb_map[x, y-i])
            i = i+1
        
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
        q.put((start_pos, 1))

        shortest_dist = deadly

        while not q.empty():
            (pos, dist) = q.get()
            # limit search depth
            if dist > s.BOMB_TIMER + 1:
                break
            # already visited
            if parents[pos] > -1:
                continue
            parents[pos] = 0
            # found safe field
            if bomb_map[pos] == 10:
                shortest_dist = dist - 1
                break

            viable = lambda x, y: (
                field[x, y] == 0 and        # field does not contain wall or crate
                (bomb_map[x, y] > dist or   # not exploded yet
                bomb_map[x, y] < dist - 1   # already faded
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
    TODO: add additional value for bomb drop action?

    :param field: Board
    :param bombs: Bomb positions
    :param explosion_map: Explosion_positions
    :param player_pos: Player position
    """
    # nothing to do if no bomb present
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

    # dropped bomb
    bomb_map = bomb_field(field, bombs + [(player_pos, 4)])
    danger[5] = danger_rating(field, bomb_map, explosion_map, player_pos, deadly)

    # not viable moves have same danger as staying
    danger[danger == -1] = danger[4]

    return danger / deadly


def crate_potential(field, player_pos):
    """
    Counts how many crates a bomb dropped at player_pos would destroy.

    :param field: Board.
    :param player_pos: Player coordinates.
    """
    crates = 0

    (x,y) = player_pos

    # go in each direction and count crates
    i = 1
    while i <= s.BOMB_POWER and field[x+i, y] != -1:
        if field[x+i, y] == 1:
            crates = crates + 1
        i = i+1
    i = 1
    while i <= s.BOMB_POWER and field[x, y+i] != -1:
        if field[x, y+i] == 1:
            crates = crates + 1
        i = i+1
    i = 1
    while i <= s.BOMB_POWER and field[x-i, y] != -1:
        if field[x-i, y] == 1:
            crates = crates + 1
        i = i+1
    i = 1
    while i <= s.BOMB_POWER and field[i, y-i] != -1:
        if field[x, y-i] == 1:
            crates = crates + 1
        i = i+1
        
    # normalize
    return np.array(crates / 4 / s.BOMB_POWER).reshape((1))

def coin_distance(field, coins, pos):
    """
    Distance to nearest reachable coin. 0 if no coin reachable.
    Pathfinding does ignore bombs and other players.

    Auxiliary for coin_collector2().

    :param field: Board
    :param coins: Coin positions
    :param player_pos: Start position
    """
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
        if pos in coins:
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

def coin_collector2(field, coins, player_pos):
    """
    Feature to find coins.
    One value for each direction.

    :param field: Board
    :param coins: Coin positions
    :param player_pos: Player position
    """
    # init return vector
    coin_rating = np.zeros(4)

    (x,y) = player_pos
    # BFS for shortest path to coin if coin exists
    if len(coins) != 0:
        if field[x+1,y] == 0:
            coin_rating[0] = coin_distance(field, coins, (x+1, y))
        if field[x,y+1] == 0:
            coin_rating[1] = coin_distance(field, coins, (x, y+1))
        if field[x-1,y] == 0:
            coin_rating[2] = coin_distance(field, coins, (x-1, y))
        if field[x,y-1] == 0:
            coin_rating[3] = coin_distance(field, coins, (x, y-1))

    # TODO evaluate whether 1 / dist is a good idea
    coin_rating[coin_rating != 0] = 1 / coin_rating[coin_rating != 0]

    return coin_rating


def coin_collector(field, coins, player_pos):
    """
    Feature designed to help collecting coins.

    One value for each direction: 1 if direction on shortest path to coin and 0 else.
    See also coin_distance().

    :param field: Board
    :param coins: Coin positions
    :param player_pos: Player position
    """
    # init return vector
    action_values = np.zeros(4)

    # BFS for shortest path to coin if coin exists
    if len(coins) >  0:
        # parent direction is given by index 0..3 counterclockwise
        dir_offset_x = [1, 0, -1, 0]
        dir_offset_y = [0, 1, 0, -1]

        # setup parent matrix
        (width, heigth) = np.shape(field)
        parents = np.full((width, heigth), -1)

        # BFS FIFO
        q = Queue()
        q.put((player_pos, 0))

        dir = -1
        while not q.empty():
            (pos,parent) = q.get()
            # already visited
            if parents[pos] > -1:
                continue
            parents[pos] = parent

            # backtrack if coin is found
            if pos in coins:
                while pos != player_pos:
                    dir = parents[pos]
                    (x, y) = pos
                    x = x - dir_offset_x[dir]
                    y = y - dir_offset_y[dir]
                    pos = (x,y)
                break

            # add accessible neighbours to queue
            (x,y) = pos
            if field[x+1, y] == 0:
                q.put(((x+1, y), 0))
            if(field[x, y+1]) == 0:
                q.put(((x, y+1), 1))
            if(field[x-1, y]) == 0:
                q.put(((x-1, y), 2))
            if(field[x, y-1]) == 0:
                q.put(((x, y-1), 3))

        # coin reachable
        if dir != -1:
            action_values[dir] = 1

    return action_values

def not_traversible(field, bombs, explosion_map, others, player_pos):
    """
    Feature informing whether adjacent fields visited (without instantly dying) or not.
    1 = no, 0 = yes

    :param field: Board
    :param bombs: bomb data
    :param explosion_map: Explosion_positions
    :param others: Status of other players
    :param player_pos: Player position
    """

    (x,y) = player_pos

    # extract other player positions
    other_pos = []
    for player in others:
        other_pos.append(player[3])

    # get bomb_field
    bomb_map = bomb_field(field, bombs)

    # conditions on traversability
    condition = lambda x,y: float(
        field[x, y] != 0 or             # field is wall or crate
        explosion_map[x, y] > 0 or      # field contains explosion
        (x,y) in other_pos or           # field contains player
        bomb_map[x,y] == 0              # field is threatened by bomb in next turn
    )

    # one value for each neighboring field
    traversible = np.zeros(4)
    traversible[0] = condition(x+1, y)
    traversible[1] = condition(x, y+1)
    traversible[2] = condition(x-1, y)
    traversible[3] = condition(x, y-1)

    return traversible

def cratos(field, player_pos):
    """
    Feature informing whether adjacent fields contain crates.
    1 = yes, 0 = no

    :param field: Board
    :param player_pos: Player position
    """

    (x,y) = player_pos

    crates = np.zeros(4)

    if field[x+1,y] == 1:
        crates[0] = 1
    if field[x-1, y] == 1:
        crates[1] = 1
    if field[x, y+1] == 1:
        crates[2] = 1
    if field[x, y-1] == 1:
        crates[4] = 1

    return crates