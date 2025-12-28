import random
from typing import Tuple, List, Any


class Snake:
    """
    Snake class representing the player's snake in the game.

    The snake consists of a body (list of coordinates) and moves in a specific direction.
    It can grow when eating food and will die if it hits walls or itself.
    """

    def __init__(self, game: Any) -> None:
        """Initialize the snake at a random position near the center of the grid."""
        self.game = game

        # Start the snake at a random position near the center
        # This prevents the snake from always starting in the exact same spot
        start_x = random.randint(game.grid_width // 2 - 5, game.grid_width // 2 + 5)
        start_y = random.randint(game.grid_height // 2 - 5, game.grid_height // 2 + 5)

        # The body is a list of (x, y) coordinates, starting with just the head
        self.body: List[Tuple[int, int]] = [(start_x, start_y)]

        # Keep track of the head position for easy access
        self.head: Tuple[int, int] = self.body[0]

        # Direction is represented as (dx, dy) - change in x and y per move
        # (0, 1) means moving down, (0, -1) means moving up
        # (1, 0) means moving right, (-1, 0) means moving left
        self.direction: Tuple[int, int] = (0, 1)  # Start moving down

        # Flag to indicate if the snake should grow on the next move
        self.grow: bool = False

    def move(self) -> None:
        """
        Move the snake forward in its current direction.

        This is the core game logic that:
        1. Calculates the new head position
        2. Checks for collisions (walls or self)
        3. Updates the snake body
        4. Handles growth
        """
        # Get current head position and direction
        x, y = self.head
        dx, dy = self.direction

        # Calculate where the new head will be
        new_head: Tuple[int, int] = (x + dx, y + dy)

        # Check for collisions
        # Collision with self: new head position is already in the body
        # Collision with walls: new head is outside the grid boundaries
        if new_head in self.body or not (
            0 <= new_head[0] < self.game.grid_width
            and 0 <= new_head[1] < self.game.grid_height
        ):
            self.game.game_over()
            return

        # Add the new head to the front of the body
        self.body.insert(0, new_head)

        # If we're not growing, remove the tail to maintain snake length
        # If we are growing, keep the tail to make the snake longer
        if not self.grow:
            self.body.pop()  # Remove the last segment (tail)
        else:
            self.grow = False  # Reset growth flag after growing

        # Update the head reference
        self.head = new_head

    def grow_snake(self) -> None:
        """
        Mark the snake to grow on the next move.

        This doesn't immediately make the snake longer - it sets a flag
        so that on the next move(), the tail won't be removed.
        """
        self.grow = True

    def change_direction(self, direction: str) -> None:
        """
        Change the snake's movement direction.

        Args:
            direction: One of "UP", "DOWN", "LEFT", "RIGHT"

        Note: The snake cannot reverse directly into itself (e.g., if moving
        right, it cannot immediately move left). The game logic should
        prevent such moves.
        """
        if direction == "UP":
            self.direction = (0, -1)
        elif direction == "DOWN":
            self.direction = (0, 1)
        elif direction == "LEFT":
            self.direction = (-1, 0)
        elif direction == "RIGHT":
            self.direction = (1, 0)

    def to_dict(self) -> List[Tuple[int, int]]:
        """
        Convert the snake to a format suitable for sending to the frontend.

        Returns:
            List of (x, y) coordinates representing the snake body
        """
        return self.body
