import datetime
import os
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearQNet(nn.Module):
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
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Any) -> Any:
        """
        Forward pass through the neural network.

        Args:
            x: Input tensor containing the game state

        Returns:
            Output tensor with Q-values for each action
        """
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self) -> None:
        """Save the trained model to disk with timestamp."""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"model_{timestamp}.pth"
        file_path = os.path.join(model_dir, file_name)
        torch.save(self.state_dict(), file_path)

    def load(self, file_name: str) -> None:
        """Load a previously saved model from disk."""
        file_path = os.path.join("models", file_name)
        self.load_state_dict(torch.load(file_path))


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
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            action_idx = int(torch.argmax(action[idx]).item())
            target[idx][action_idx] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
