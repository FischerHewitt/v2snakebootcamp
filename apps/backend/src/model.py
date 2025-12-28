# TODO: Import PyTorch for building neural networks
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import datetime
from typing import Any


class LinearQNet:
    """
    A simple neural network for Q-learning in the Snake game.

    This is a basic feedforward neural network with:
    - Input layer: game state features (13 inputs)
    - Hidden layer: fully connected layer with ReLU activation
    - Output layer: Q-values for each action (3 outputs: straight, right, left)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the neural network layers.

        Args:
            input_size: Number of input features (13 for snake game state)
            hidden_size: Number of neurons in hidden layer (e.g., 256)
            output_size: Number of output actions (3: straight, right, left)
        """
        # Initialize the neural network as a PyTorch nn.Module
        super().__init__()

        # TODO: Create the network layers

        pass

    def forward(self, x: Any) -> Any:
        """
        Forward pass through the neural network.

        Args:
            x: Input tensor containing the game state

        Returns:
            Output tensor with Q-values for each action
        """
        # TODO: Apply ReLU activation to first layer
        # TODO: Apply second layer (no activation for Q-values)

        return x

    def save(self) -> None:
        """Save the trained model to disk with timestamp."""
        # TODO: Create model directory if it doesn't exist
        # TODO: Generate filename with timestamp
        # TODO: Save the model state dictionary

        pass

    def load(self, file_name: str) -> None:
        """Load a previously saved model from disk."""
        # TODO: Construct full file path
        # TODO: Load the model state dictionary

        pass


class QTrainer:
    """
    Trainer class for the Q-learning neural network.

    Handles the training process using the Bellman equation:
    Q(s,a) = r + γ * max(Q(s',a'))

    Where:
    - Q(s,a) = Q-value for state s and action a
    - r = immediate reward
    - γ = discount factor (gamma)
    - s' = next state
    - a' = possible actions in next state
    """

    def __init__(self, model: Any, lr: float, gamma: float) -> None:
        """
        Initialize the trainer with model and hyperparameters.

        Args:
            model: The neural network to train
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
        """
        # TODO: Store hyperparameters
        # TODO: Initialize Adam optimizer
        # TODO: Initialize Mean Squared Error loss function

        pass

    def train_step(
        self, state: Any, action: Any, reward: Any, next_state: Any, done: Any
    ) -> None:
        """
        Perform one training step on the neural network.

        This implements the Q-learning algorithm update rule.

        Args:
            state: Current game state(s)
            action: Action(s) taken
            reward: Reward(s) received
            next_state: Next game state(s)
            done: Whether the game ended
        """
        # TODO: Handle both single experiences and batches
        # TODO: Get current Q-values from the model
        # TODO: Clone predictions to create target values
        # TODO: Update target values using Bellman equation
        # TODO: Perform gradient descent

        pass
