import numpy as np
import random
import copy

class Game2048:
    def __init__(self, max_moves=200):
        self.max_moves = max_moves
        self.moves_count = 0
        self.previous_score = 0  # Initialize previous_score
        self.reset()

    def reset(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.moves_count = 0
        self.previous_score = 0  # Reset previous_score
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()

    def add_random_tile(self):
        empty_positions = list(zip(*np.where(self.grid == 0)))
        if empty_positions:
            i, j = random.choice(empty_positions)
            self.grid[i, j] = 2 if random.random() < 0.9 else 4

    def slide_left(self, matrix):
        new_matrix = np.zeros_like(matrix)
        reward = 0

        for i in range(4):
            pos = 0
            for j in range(4):
                if matrix[i, j] != 0:
                    value = matrix[i, j]
                    if pos > 0 and new_matrix[i, pos - 1] == value:
                        new_matrix[i, pos - 1] *= 2
                        self.score += new_matrix[i, pos - 1]
                    else:
                        new_matrix[i, pos] = value
                        pos += 1
        return new_matrix, reward

    def move(self, direction):
        if direction == 'left':
            new_grid, _ = self.slide_left(self.grid)
        elif direction == 'right':
            self.grid = np.fliplr(self.grid)
            new_grid, _ = self.slide_left(self.grid)
            new_grid = np.fliplr(new_grid)
        elif direction == 'up':
            self.grid = self.grid.T
            new_grid, _ = self.slide_left(self.grid)
            new_grid = new_grid.T
        elif direction == 'down':
            self.grid = self.grid.T
            new_grid, _ = self.slide_left(self.grid)
            new_grid = np.fliplr(new_grid.T).T
        else:
            raise ValueError("Invalid move direction")

        if not np.array_equal(self.grid, new_grid):
            self.grid = new_grid
            self.add_random_tile()
            self.moves_count += 1
            return self.get_reward(), self.is_game_over()
        else:
            return -1, False

    def is_game_over(self):
        if np.any(self.grid == 0):
            return False

        for i in range(4):
            for j in range(3):
                if (self.grid[i, j] == self.grid[i, j + 1]) or (self.grid[j, i] == self.grid[j + 1, i]):
                    return False
        return True

    def get_state(self):
        return copy.deepcopy(self.grid).flatten(), self.score

    def get_reward(self):
        reward = 0

        # Strategy: Largest tile should be in the top right corner
        largest_tile = np.max(self.grid)
        if self.grid[0, 3] == largest_tile:
            reward += 5  # Reward for keeping the largest tile in the top right corner

        # Penalize if the largest tile is not in the top right corner
        if self.grid[0, 3] != largest_tile:
            reward -= 10

        # Reward for score increases
        reward += self.score - self.previous_score
        self.previous_score = self.score  # Update for the next move

        # Penalize if move limit is reached
        if self.moves_count >= self.max_moves:
            reward -= 10

        return reward


    def step(self, action):
        initial_score = self.score
        if action == 0:
            move_direction = 'left'
            reward, done = self.move(move_direction)
        elif action == 1:
            move_direction = 'right'
            reward, done = self.move(move_direction)
        elif action == 2:
            move_direction = 'up'
            reward, done = self.move(move_direction)
        elif action == 3:
            move_direction = 'down'
            reward, done = self.move(move_direction)
        else:
            raise ValueError("Invalid action")

        new_state, _ = self.get_state()
        
        #to be uncommented if needed to print board
        #self.print_board()

        return new_state, reward, done

    def print_board(self):
        print("\nCurrent Board:")
        for row in self.grid:
            print(" ".join(map(lambda x: f"{x:4d}", row)))
        print(f"Score: {self.score}\n")
