from typing import Deque, List, Optional

# Import necessary libraries
import torch
import torch.nn as nn
from collections import deque
import random
from game import Game

from model import LinearQNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001
GAMMA = 0.9
EPSILON_START = 80


class DQN:
    """
    Deep Q-Network agent for playing Snake using reinforcement learning.

    This agent uses a neural network to learn the optimal policy for playing Snake.
    It learns through trial and error, getting rewards for good actions (eating food)
    and penalties for bad actions (hitting walls or itself).
    """

    def __init__(self: "DQN") -> None:
        """Initialize the DQN agent with all necessary components."""
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory: Deque = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(13, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.prev_distance: Optional[float] = None

    def get_state(self, game: "Game") -> List[float]:
        """
        Extract the current state of the game as input features for the neural network.

        The state includes:
        - Danger detection in three directions (straight, right, left)
        - Food direction relative to snake head (up, down, left, right)
        - Normalized distances to food
        - Current snake direction
        """
        head_x, head_y = game.snake.head
        food_x, food_y = game.food.position

        def is_collision(point: tuple[int, int]) -> bool:
            x, y = point
            if x < 0 or x >= game.grid_width or y < 0 or y >= game.grid_height:
                return True
            return point in game.snake.body

        dir_left = game.snake.direction == (-1, 0)
        dir_right = game.snake.direction == (1, 0)
        dir_up = game.snake.direction == (0, -1)
        dir_down = game.snake.direction == (0, 1)

        if dir_right:
            point_straight = (head_x + 1, head_y)
            point_right = (head_x, head_y + 1)
            point_left = (head_x, head_y - 1)
        elif dir_left:
            point_straight = (head_x - 1, head_y)
            point_right = (head_x, head_y - 1)
            point_left = (head_x, head_y + 1)
        elif dir_up:
            point_straight = (head_x, head_y - 1)
            point_right = (head_x + 1, head_y)
            point_left = (head_x - 1, head_y)
        else:
            point_straight = (head_x, head_y + 1)
            point_right = (head_x - 1, head_y)
            point_left = (head_x + 1, head_y)

        danger_straight = is_collision(point_straight)
        danger_right = is_collision(point_right)
        danger_left = is_collision(point_left)

        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y

        dx_norm = (food_x - head_x) / max(game.grid_width, 1)
        dy_norm = (food_y - head_y) / max(game.grid_height, 1)

        state = [
            float(danger_straight),
            float(danger_right),
            float(danger_left),
            float(dir_left),
            float(dir_right),
            float(dir_up),
            float(dir_down),
            float(food_left),
            float(food_right),
            float(food_up),
            float(food_down),
            dx_norm,
            dy_norm,
        ]
        return state

    def calculate_reward(self, game: "Game", done: bool) -> int:
        """
        Calculate the reward for the current game state.

        Rewards encourage good behavior:
        - Positive reward for eating food
        - Small positive reward for moving closer to food
        - Small negative reward for moving away from food
        - Large negative reward for dying
        """
        if done:
            self.prev_distance = None
            return -10

        head_x, head_y = game.snake.head
        food_x, food_y = game.food.position
        distance = abs(food_x - head_x) + abs(food_y - head_y)

        reward = 0
        if self.prev_distance is not None:
            reward += 1 if distance < self.prev_distance else -1

        if getattr(game.snake, "grow", False):
            reward += 10

        self.prev_distance = distance
        return reward

    def remember(
        self,
        state: List[float],
        action: List[int],
        reward: int,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Store an experience in memory for later training (experience replay)."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Train the neural network on a batch of experiences from memory."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        if not mini_sample:
            return

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(
        self,
        state: List[float],
        action: List[int],
        reward: int,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Train the neural network on a single experience (immediate learning)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: List[float]) -> List[int]:
        """
        Choose an action based on the current state.

        Uses epsilon-greedy strategy:
        - With probability epsilon: choose random action (exploration)
        - With probability 1-epsilon: choose best action from neural network (exploitation)

        Actions: [1,0,0] = straight, [0,1,0] = turn right, [0,0,1] = turn left
        """
        self.epsilon = max(0, EPSILON_START - self.n_games)
        final_move = [0, 0, 0]
        if random.random() < (self.epsilon / 200):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1
        return final_move
