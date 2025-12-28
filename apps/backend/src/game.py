from snake import Snake
from food import Food
import time
from typing import List, Dict, Any


class Game:
    """
    Main game class that manages the Snake game state and logic.

    This class coordinates the snake, food, scoring, and game timing.
    It serves as the central controller for the entire game.
    """

    def __init__(self) -> None:
        """Initialize a new game with default settings."""
        # Grid dimensions (in cells, not pixels)
        self.grid_width: int = 29  # Number of cells horizontally
        self.grid_height: int = 19  # Number of cells vertically

        # Game state
        self.score: int = 0  # Current score (increases when eating food)
        self.running: bool = True  # Whether the game is still active

        # Game objects
        self.snake: Snake = Snake(self)  # The player's snake
        self.food: Food = Food(self)  # The food to collect

        # Timing control
        self.game_tick: float = 0.03  # Time between updates (seconds)
        self.last_tick: float = time.time()  # When the last update occurred

        # Input queue to handle multiple rapid inputs
        self.change_queue: List[Any] = []

    def game_over(self) -> None:
        """
        End the current game.

        Called when the snake hits a wall or itself.
        Sets the running flag to False to stop the game loop.
        """
        self.running = False

    def send(self) -> Dict[str, Any]:
        """
        Create a dictionary representation of the current game state.

        This is used for sending game updates to clients.

        Returns:
            Dictionary containing all relevant game information
        """
        event: Dict[str, Any] = {
            "score": self.score,
            "tick": self.game_tick,
            "snake": self.snake.body,  # List of snake body coordinates
            "food": self.food.position,  # Food position (x, y)
            "running": self.running,  # Whether game is still active
        }
        return event

    def step(self) -> None:
        """
        Advance the game by one frame.

        This is the main game update function that:
        1. Processes any queued direction changes
        2. Moves the snake
        3. Checks if food was eaten
        4. Updates the timing
        """
        # Don't update if the game is over
        if not self.running:
            return

        # Process any direction changes from the input queue
        if len(self.change_queue) > 0:
            self.snake.change_direction(self.change_queue.pop())
            self.change_queue = []  # Clear the queue after processing

        # Move the snake forward one step
        self.snake.move()

        # Check if the snake ate the food
        self.food.check_eaten()

        # Update timing for next frame
        self.last_tick = time.time()

    def queue_change(self, update: Any) -> None:
        """
        Queue a direction change to be processed on the next game step.

        This prevents the snake from changing direction multiple times
        in a single frame, which could cause it to reverse into itself.

        Args:
            update: Direction string ("UP", "DOWN", "LEFT", "RIGHT")
        """
        if not self.running:
            return
        self.change_queue.append(update)

    def reset(self) -> None:
        """
        Reset the game to its initial state for a new round.

        This creates a new snake and food, resets the score,
        and starts the game running again.
        """
        self.score = 0
        self.snake = Snake(self)
        self.food = Food(self)
        self.running = True
        self.last_tick = time.time()

    def to_vector(self) -> List[int]:
        """
        Convert key game state information to a simple list format.

        This is useful for AI agents that need numerical input.

        Returns:
            List containing [snake_head_x, snake_head_y, food_x, food_y, direction_x, direction_y]
        """
        return [
            self.snake.head[0],  # Snake head X coordinate
            self.snake.head[1],  # Snake head Y coordinate
            self.food.position[0],  # Food X coordinate
            self.food.position[1],  # Food Y coordinate
            self.snake.direction[0],  # Snake direction X component
            self.snake.direction[1],  # Snake direction Y component
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire game state to a dictionary.

        This is the complete game state suitable for serialization
        and sending to frontend clients.

        Returns:
            Dictionary containing all game state information
        """
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "game_tick": self.game_tick,
            "snake": self.snake.to_dict(),  # Snake body coordinates
            "food": self.food.to_dict(),  # Food position
            "score": self.score,
        }
