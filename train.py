# import numpy as np
# import random
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from collections import deque
# from game_envir import Game2048

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=1000)  # Memory size of 1000
#         self.gamma = 0.95  # Discount rate
#         self.epsilon = 1.0  # Exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.996  # Epsilon decay rate
#         self.learning_rate = 0.001
#         self.model = self.build_model()

#     def build_model(self):
#         # Neural network model for Q-learning
#         model = Sequential([
#             Dense(64, input_dim=self.state_size, activation='relu'),
#             Dense(64, activation='relu'),
#             Dense(self.action_size, activation='linear')
#         ])
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model(state, training=False)
#         return tf.argmax(act_values[0]).numpy()

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def save(self, name):
#         self.model.save_weights(name)

#     def load(self, name):
#         self.model.load_weights(name)

# if __name__ == "__main__":
#     # Initialize the game environment and agent
#     env = Game2048()  # Set max_moves to 1 for testing
#     state_size = 16  # 4x4 board flattened
#     action_size = 4  # Four possible moves: left, right, up, down
#     agent = DQNAgent(state_size, action_size)

#     episodes = 1  # Total number of episodes
#     batch_size = 1  # Size of minibatch for replay

#     for e in range(episodes):
#         print(f"\rEpisode: {e + 1}/{episodes}", end="")  # Continuously print the current episode

#         state, score = env.reset()
#         state = np.reshape(state, [1, state_size])

#         for time in range(500):  # Limit the maximum number of steps per episode
#             action = agent.act(state)
#             next_state, reward, done = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, state_size])

#             agent.remember(state, action, reward, next_state, done)
#             state = next_state

#             if done:
#                 print(f"\nEpisode: {e + 1}/{episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")
#                 break  # Exit loop when done is True

#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)

#         # Save the model weights every 10 episodes
#         if e % 10 == 0:
#             agent.save(f"models/2048-model-lvl-{e}.weights.h5")

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
from game_envir import Game2048

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # Memory size of 1000
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996  # Epsilon decay rate
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        # Neural network model for Q-learning
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

if __name__ == "__main__":
    # Initialize the game environment and agent
    env = Game2048()  # Set max_moves to 1 for testing
    state_size = 16  # 4x4 board flattened
    action_size = 4  # Four possible moves: left, right, up, down
    agent = DQNAgent(state_size, action_size)

    episodes = 200  # Total number of episodes
    batch_size = 32  # Size of minibatch for replay

    for e in range(episodes):
        print(f"\rEpisode: {e + 1}/{episodes}", end="")  # Continuously print the current episode

        state, score = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):  # Limit the maximum number of steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"\nEpisode: {e + 1}/{episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")
                break  # Exit loop when done is True

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Save the model weights every 10 episodes
        if e % 10 == 0:
            agent.save(f"models/2048-model-lvl-{e}.weights.h5")
