from typing import Deque, Tuple, List, Optional

# Import necessary libraries
import torch
import torch.nn as nn
from collections import deque
import random
from game import Game

from model import LinearQNet, QTrainer


# TODO: Define constants for the DQN agent


class DQN:
    """
    Deep Q-Network agent for playing Snake using reinforcement learning.

    This agent uses a neural network to learn the optimal policy for playing Snake.
    It learns through trial and error, getting rewards for good actions (eating food)
    and penalties for bad actions (hitting walls or itself).
    """

    def __init__(self: "DQN") -> None:
        """Initialize the DQN agent with all necessary components."""
        # TODO: Initialize training statistics
        # TODO: Initialize epsilon-greedy exploration parameters
        # TODO: Initialize memory for experience replay
        # TODO: Initialize the neural network and trainer

        pass

    def get_state(self, game: "Game") -> List[float]:
        """
        Extract the current state of the game as input features for the neural network.

        The state includes:
        - Danger detection in three directions (straight, right, left)
        - Food direction relative to snake head (up, down, left, right)
        - Normalized distances to food
        - Current snake direction
        """
        # TODO: Get the snake's head position
        # TODO: Helper function to normalize distances
        # TODO: Get current direction as one-hot encoding

        # TODO: Detect dangers in three directions relative to current direction
        # TODO: Get food direction relative to snake head
        # TODO: Calculate normalized distances to food
        # TODO: Combine all features into a single state vector
        # TODO: Return state as PyTorch tensor

        return []  # Placeholder return for now

    def calculate_reward(self, game: "Game", done: bool) -> int:
        """
        Calculate the reward for the current game state.

        Rewards encourage good behavior:
        - Positive reward for eating food
        - Small positive reward for moving closer to food
        - Small negative reward for moving away from food
        - Large negative reward for dying
        """
        # TODO: Initialize reward
        # TODO: Get current positions
        # TODO: Calculate distance-based rewards
        # TODO: Big reward for eating food
        # TODO: Big penalty for dying

        return 0

    def remember(
        self,
        state: List[float],
        action: List[int],
        reward: int,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Store an experience in memory for later training (experience replay)."""
        # TODO: Add the experience to memory

        pass

    def train_long_memory(self) -> None:
        """Train the neural network on a batch of experiences from memory."""
        # TODO: Sample a batch of experiences from memory
        # TODO: Unpack the batch and train the model

        pass

    def train_short_memory(
        self,
        state: List[float],
        action: List[int],
        reward: int,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Train the neural network on a single experience (immediate learning)."""
        pass

    def get_action(self, state: List[float]) -> List[int]:
        """
        Choose an action based on the current state.

        Uses epsilon-greedy strategy:
        - With probability epsilon: choose random action (exploration)
        - With probability 1-epsilon: choose best action from neural network (exploitation)

        Actions: [1,0,0] = straight, [0,1,0] = turn right, [0,0,1] = turn left
        """
        # TODO: Decay epsilon over time (explore less as agent learns)
        # TODO: Initialize action array
        # TODO: Epsilon-greedy action selection

        return [1, 0, 0]
