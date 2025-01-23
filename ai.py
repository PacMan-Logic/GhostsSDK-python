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

        for ghost_id in ghost_order:
            current_pos = game_state.ghosts_pos[ghost_id]
            valid_moves = self.get_valid_moves(current_pos, game_state)

            if not valid_moves:
                moves[ghost_id] = Direction.STAY.value
                continue

            best_move = random.choice(valid_moves)
            self.update_history(ghost_id, best_move[0])
            moves[ghost_id] = best_move[1]
            continue

        return moves

# TODO: 你需要实现一个ai函数

ai_func = GhostAI().choose_moves # TODO: 你需要把ai_func替换为自己的ai函数
__all__ = ["ai_func"]
