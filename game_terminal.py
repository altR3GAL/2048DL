import numpy as np
import random

class Game2048:
    def __init__(self, max_moves=50):
        self.max_moves = max_moves
        self.moves_count = 0
        self.previous_score = 0
        self.reset()

    def reset(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.moves_count = 0
        self.previous_score = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty_positions = list(zip(*np.where(self.grid == 0)))
        if empty_positions:
            i, j = random.choice(empty_positions)
            self.grid[i, j] = 2 if random.random() < 0.9 else 4

    def slide_left(self):
        new_grid = np.zeros_like(self.grid)
        reward = 0

        for i in range(4):
            pos = 0
            for j in range(4):
                if self.grid[i, j] != 0:
                    value = self.grid[i, j]
                    if pos > 0 and new_grid[i, pos - 1] == value:
                        new_grid[i, pos - 1] *= 2
                        self.score += new_grid[i, pos - 1]
                        reward += new_grid[i, pos - 1]
                    else:
                        new_grid[i, pos] = value
                        pos += 1
        return new_grid, reward

    def move(self, direction):
        if direction == 'left':
            new_grid, reward = self.slide_left()
        elif direction == 'right':
            self.grid = np.fliplr(self.grid)
            new_grid, reward = self.slide_left()
            new_grid = np.fliplr(new_grid)
        elif direction == 'up':
            self.grid = self.grid.T
            new_grid, reward = self.slide_left()
            new_grid = new_grid.T
        elif direction == 'down':
            self.grid = self.grid.T
            new_grid, reward = self.slide_left()
            new_grid = np.fliplr(new_grid.T).T
        else:
            raise ValueError("Invalid move direction")

        if not np.array_equal(self.grid, new_grid):
            self.grid = new_grid
            self.add_random_tile()
            self.moves_count += 1
            return reward, self.is_game_over()
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

    def get_reward(self):
        reward = 0
        largest_tile = np.max(self.grid)
        if self.grid[0, 3] == largest_tile:
            reward += 5
        else:
            reward -= 10
        reward += self.score - self.previous_score
        self.previous_score = self.score
        if self.moves_count >= self.max_moves:
            reward -= 10
        return reward

    def step(self, action):
        directions = ['up', 'left', 'down', 'right']
        reward, done = self.move(directions[action])
        return self.grid.flatten(), reward, done

    def print_board(self):
        for row in self.grid:
            print("\t".join(str(x) if x != 0 else '.' for x in row))
        print(f"Score: {self.score}")

if __name__ == "__main__":
    game = Game2048()
    while not game.is_game_over():
        game.print_board()
        move = input("Enter move (w, a, s, d): ").strip().lower()
        if move in ['w','a', 's', 'd']:
            move_map = {'w': 0, 'a': 1, 's': 2, 'd': 3}
            _, reward, done = game.step(move_map[move])
            print(f"Reward: {reward}")
            if done:
                break
        else:
            print("Invalid move. Please enter 'w', 'a', 's', or 'd'.")
    game.print_board()
    print("Game Over!")
