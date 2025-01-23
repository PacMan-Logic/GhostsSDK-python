from core.gamedata import *
import random
import numpy as np



def parse(x: tuple):
    if x == (0, 0):
        return Direction.STAY.value
    if x == (1, 0):
        return Direction.UP.value
    if x == (-1, 0):
        return Direction.DOWN.value
    if x == (0, 1):
        return Direction.RIGHT.value
    if x == (0, -1):
        return Direction.LEFT.value


class GhostAI:
    def __init__(self):
        self.position_history = {0: [], 1: [], 2: []}
        self.history_length = 5

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_valid_moves(self, pos, game_state):
        valid_moves = []
        directions = [
            ([1, 0], 1),  # UP
            ([-1, 0], 3),  # DOWN
            ([0, -1], 2),  # LEFT
            ([0, 1], 4),  # RIGHT
        ]

        for direction, move_value in directions:
            new_pos = [pos[0] + direction[0], pos[1] + direction[1]]
            if (
                0 <= new_pos[0] < game_state.board_size
                and 0 <= new_pos[1] < game_state.board_size
                and game_state.board[new_pos[0]][new_pos[1]] != 0
            ):
                valid_moves.append((new_pos, move_value))
        return valid_moves

    def a_star_search(self, start: np.ndarray, goal: np.ndarray, game_state: GameState):
        open_set = set()
        open_set.add(tuple(start))
        came_from = {}

        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.manhattan_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for direction, _ in self.get_valid_moves(list(current), game_state):
                neighbor = tuple(direction)
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []

    def calculate_stagnation_penalty(self, new_pos, ghost_id):
        if not self.position_history[ghost_id]:
            return 0
        repeat_count = sum(
            1
            for pos in self.position_history[ghost_id]
            if pos[0] == new_pos[0] and pos[1] == new_pos[1]
        )
        return repeat_count * 2

    def update_history(self, ghost_id, new_pos):
        self.position_history[ghost_id].append(new_pos)
        if len(self.position_history[ghost_id]) > self.history_length:
            self.position_history[ghost_id].pop(0)

    def get_map_distance(self, ghost_id, pacman_position: np.ndarray, ghost_positions: list[np.ndarray], game_state: GameState):
        if ghost_id == -1:
            pos = pacman_position
        else:
            pos = ghost_positions[ghost_id]

        # bfs
        queue = [pos]
        visited = set()
        visited.add(tuple(pos))
        distance = 0
        map_distance = np.zeros((game_state.board_size, game_state.board_size))
        while queue:
            distance += 1
            for _ in range(len(queue)):
                pos = queue.pop(0)
                for new_pos, _ in self.get_valid_moves(pos, game_state):
                    if tuple(new_pos) not in visited:
                        queue.append(new_pos)
                        visited.add(tuple(new_pos))
                        map_distance[new_pos[0], new_pos[1]] = distance
        return map_distance
    
    def get_pacman_controlled_area(self, pacman_position, ghost_positions, game_state: GameState):
        ghost_map_distance = [self.get_map_distance(ghost_id, pacman_position, ghost_positions, game_state) for ghost_id in range(3)]
        pacman_map_distance = self.get_map_distance(-1, pacman_position, ghost_positions, game_state)
        controlled_area = np.zeros((game_state.board_size, game_state.board_size))
        for x in range(game_state.board_size):
            for y in range(game_state.board_size):
                if pacman_map_distance[x, y] < min([ghost_map_distance[ghost_id][x, y] for ghost_id in range(3)]):
                    controlled_area[x, y] = 1
        return controlled_area

    def choose_moves(self, game_state: GameState):
        moves = [0, 0, 0]
        pacman_pos = game_state.pacman_pos

        ghost_distances = [float("inf"), float("inf"), float("inf")]
        ghost_order = [0, 1, 2]

        for ghost_id in range(3):
            # 计算到吃豆人的距离
            current_pos = game_state.ghosts_pos[ghost_id]
            a_star_path = self.a_star_search(current_pos, pacman_pos, game_state)
            ghost_distances[ghost_id] = len(a_star_path) if a_star_path else float("inf")

        ghost_order = sorted(ghost_order, key=lambda x: ghost_distances[x])

        distance_1_flag = False # 是否已经有一个ghost距离为1

        for ghost_id in ghost_order:
            current_pos = game_state.ghosts_pos[ghost_id]
            valid_moves = self.get_valid_moves(current_pos, game_state)

            if not valid_moves:
                moves[ghost_id] = Direction.STAY.value
                continue

            if ghost_distances[ghost_id] == 1:
                if distance_1_flag and len(valid_moves) > 1:
                    # 如果已经有一个ghost距离为1，且当前ghost有多个可选方向，则避免两个ghost重叠
                    candidate_moves = [move for move in valid_moves if move[0][0] != pacman_pos[0] or move[0][1] != pacman_pos[1]]
                else:
                    # 否则直接追吃豆人
                    distance_1_flag = True
                    moves[ghost_id] = parse((pacman_pos[0] - current_pos[0], pacman_pos[1] - current_pos[1]))
                    continue

            else:
                candidate_moves = valid_moves.copy()

            # 考虑最大程度限制吃豆人map_distance小于ghost_map_distance的格子数量
            min_controlled_area = float("inf")
            min_distance = float("inf")
            best_move = []

            for(new_pos, _) in candidate_moves:
                new_ghost_positions = game_state.ghosts_pos.copy()
                new_ghost_positions[ghost_id] = np.array(new_pos)
                controlled_area = np.sum(self.get_pacman_controlled_area(pacman_pos, new_ghost_positions, game_state))
                if controlled_area < min_controlled_area:
                    best_move.clear()
                    best_move.append(new_pos)
                    min_controlled_area = controlled_area
                    min_distance = len(self.a_star_search(new_pos, pacman_pos, game_state))
                elif controlled_area == min_controlled_area:
                    new_distance = len(self.a_star_search(new_pos, pacman_pos, game_state))
                    if new_distance < min_distance:
                        best_move.clear()
                        best_move.append(new_pos)
                        min_distance = new_distance
                    elif new_distance == min_distance:
                        best_move.append(new_pos)
                        
            if len(best_move) == 1:
                moves[ghost_id] = parse((best_move[0][0] - current_pos[0], best_move[0][1] - current_pos[1]))
            else:
                random_move = random.choice(best_move)
                moves[ghost_id] = parse((random_move[0] - current_pos[0], random_move[1] - current_pos[1]))            

        return moves

ai_func = GhostAI().choose_moves
__all__ = ["ai_func"]
