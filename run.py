import numpy as np
from game_env import Game2048
from train import DQNAgent  # Assuming DQNAgent is defined in train_2048.py

if __name__ == "__main__":
    env = Game2048()
    state_size = 16  # 4x4 board flattened
    action_size = 4  # left, right, up, down
    agent = DQNAgent(state_size, action_size)

    # Load the trained model
    agent.load("2048-dqn-990.weights.h5")  # Replace with your saved model's filename

    # Watch the trained bot play 2048
    agent.play_game(episodes=5)
