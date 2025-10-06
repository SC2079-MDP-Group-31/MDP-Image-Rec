from typing import Dict, Tuple

from commands.command import Command
from misc.direction import Direction
from misc.positioning import Position, RobotPosition
from misc.type_of_turn import TypeOfTurn


class TurnCommand(Command):
    # Time constants for turn type
    TURN_TIMES = {
        TypeOfTurn.MEDIUM: 30,  # SOME VALUE TO BE EMPIRICALLY DETERMINED
    }

    # Position deltas for different turn combinations
    MEDIUM_TURN_DELTAS = {
        # (left, right, reverse): {direction: (dx, dy, new_direction)}
        (True, False, False): {  # Left forward
            Direction.TOP: (-20, 30, Direction.LEFT),
            Direction.LEFT: (-30, -20, Direction.BOTTOM),
            Direction.RIGHT: (30, 20, Direction.TOP),
            Direction.BOTTOM: (20, -30, Direction.RIGHT),
        },
        (False, True, False): {  # Right forward
            Direction.TOP: (20, 30, Direction.RIGHT),
            Direction.LEFT: (-30, 20, Direction.TOP),
            Direction.RIGHT: (30, -20, Direction.BOTTOM),
            Direction.BOTTOM: (-20, -30, Direction.LEFT),
        },
        (True, False, True): {  # Left reverse
            Direction.TOP: (-30, -20, Direction.RIGHT),
            Direction.LEFT: (20, -30, Direction.TOP),
            Direction.RIGHT: (-20, 30, Direction.BOTTOM),
            Direction.BOTTOM: (30, 20, Direction.LEFT),
        },
        (False, True, True): {  # Right reverse
            Direction.TOP: (30, -20, Direction.LEFT),
            Direction.LEFT: (20, 30, Direction.BOTTOM),
            Direction.RIGHT: (-20, -30, Direction.TOP),
            Direction.BOTTOM: (-30, 20, Direction.RIGHT),
        },
    }

    # Command messages for different turn combinations
    COMMAND_MESSAGES = {
        # (left, right, reverse, turn_type): message
        (True, False, False, TypeOfTurn.MEDIUM): "FL090",  # turn left medium forward
        (True, False, True, TypeOfTurn.MEDIUM): "LB090",  # turn left medium reverse
        (False, True, False, TypeOfTurn.MEDIUM): "FR090",  # turn right medium forward
        (False, True, True, TypeOfTurn.MEDIUM): "RB090",  # turn right medium reverse
    }

    def __init__(self, type_of_turn: TypeOfTurn, left: bool, right: bool, reverse: bool):
        """
        Initialize a turn command.

        Args:
            type_of_turn: The magnitude of the turn (only MEDIUM supported)
            left: Whether to turn the front wheels left
            right: Whether to turn the front wheels right
            reverse: Whether to execute the turn in reverse
        """
        time = self.TURN_TIMES.get(type_of_turn, 0)
        super().__init__(time)

        self.type_of_turn = type_of_turn
        self.left = left
        self.right = right
        self.reverse = reverse

    def __str__(self) -> str:
        return f"TurnCommand:{self.type_of_turn}, rev={self.reverse}, left={self.left}, right={self.right})"

    __repr__ = __str__

    def process_one_tick(self, robot) -> None:
        """Process one tick of the turn command."""
        if self.total_ticks == 0:
            return

        self.tick()
        robot.turn(self.type_of_turn, self.left, self.right, self.reverse)

    def get_type_of_turn(self) -> TypeOfTurn:
        """Get the type of turn."""
        return self.type_of_turn

    def _get_position_delta(self, direction: Direction) -> Tuple[int, int, Direction]:
        """
        Get the position delta for the current turn configuration and direction.

        Args:
            direction: Current robot direction

        Returns:
            Tuple of (dx, dy, new_direction). new_direction is None if unchanged.
        """
        turn_key = (self.left, self.right, self.reverse)
        deltas = self.MEDIUM_TURN_DELTAS
        return deltas.get(turn_key, {}).get(direction, (0, 0, None))

    def apply_on_pos(self, curr_pos: Position) -> 'TurnCommand':
        """
        Apply the turn command to the current robot position.

        Args:
            curr_pos: Current robot position

        Returns:
            Self for method chaining
        """
        if not isinstance(curr_pos, RobotPosition):
            raise ValueError("Cannot apply turn command on non-robot positions!")

        dx, dy, new_direction = self._get_position_delta(curr_pos.direction)

        # Apply position changes
        curr_pos.x += dx
        curr_pos.y += dy

        # Update direction if it changed
        if new_direction is not None:
            curr_pos.direction = new_direction

        return self

    def convert_to_message(self) -> str:
        """
        Convert the turn command to a message string for robot communication.

        Returns:
            Command message string

        Possible command messages:
        # Left forward turns
        #   MEDIUM: "FL090"  (turn left medium forward)
        # Left reverse turns
        #   MEDIUM: "LB090"  (turn left medium reverse)
        # Right forward turns
        #   MEDIUM: "FR090"  (turn right medium forward)
        # Right reverse turns
        #   MEDIUM: "RB090"  (turn right medium reverse)
        # If configuration is not recognized, returns "UNKNOWN_COMMAND".
        """
        command_key = (self.left, self.right, self.reverse, self.type_of_turn)
        return self.COMMAND_MESSAGES.get(command_key, "UNKNOWN_COMMAND")