import sys
import time
from typing import List
import constants
from commands.go_straight_command import StraightCommand
from commands.scan_obstacle_command import ScanCommand
from grid.grid import Grid
from grid.obstacle import Obstacle
from misc.direction import Direction
from misc.positioning import Position
from pygame_app import AlgoMinimal
from robot.robot import Robot

def run_minimal(raw_data):
    also_run_simulator = False  # Set to False if you don't want to run the simulator
    if isinstance(raw_data, bytes):
        decoded_data = raw_data.decode("utf-8")
    else:
        decoded_data = raw_data
        
    print(f"Received: {decoded_data}")

    if decoded_data.startswith("ALG:"):
        obstacle_data = parse_rpi_message(decoded_data)
        print(f"Parsed obstacle data: {obstacle_data}")
        commands = process_obstacle_data(obstacle_data, also_run_simulator)
        return commands
    else:
        commands = process_string_command(decoded_data)
        return commands


# ------------------ Helper Functions -----------------------------------
def parse_rpi_message(message: str) -> List[List[int]]:
        """
        Parse message from RPI in format: ALG:x,y,direction,id;x,y,direction,id;...

        Args:
            message: Raw message string from RPI

        Returns:
            Parsed obstacle data
        """
        # Remove ALG: prefix and split by semicolon
        data = message[4:].split(";")[:-1]  # Last element is empty string
        parsed_obstacles = []

        for obstacle_str in data:
            parts = obstacle_str.split(",")
            if len(parts) != 4:
                continue

            # Convert coordinates (multiply by 10 for correct scale)
            x = int(parts[0]) * 10
            y = int(parts[1]) * 10

            # Convert direction
            direction_map = {"N": 90, "S": -90, "E": 0, "W": 180}
            direction = direction_map.get(parts[2], 0)

            obstacle_id = int(parts[3])
            parsed_obstacles.append([x, y, direction, obstacle_id])

        return parsed_obstacles


def process_obstacle_data(data: List[List[int]], also_run_simulator: bool):
    """Process obstacle data and execute algorithm."""
    obstacles = parse_obstacle_data(data)
    app = AlgoMinimal(obstacles)

    app.execute()

    # Get path planning results
    obs_priority = app.robot.hamiltonian.get_simple_hamiltonian()

    # Convert to commands and send to RPI
    commands = app.robot.convert_commands_to_messages()
    print(f"Commands to send: {commands}")

    if commands:
        return commands
    else:
        print("ERROR!! NO COMMANDS TO SEND TO RPI")

def parse_obstacle_data(data: List[List]) -> List[Obstacle]:
    """
    Parse obstacle data from the format [[x, y, orient, index], ...] into Obstacle objects.

    Args:
        data: List of obstacle parameters [x, y, orientation, index]

    Returns:
        List of Obstacle objects
    """
    obstacles = []
    for obstacle_params in data:
        if len(obstacle_params) < 4:
            continue

        obstacle = Obstacle(
            Position(
                obstacle_params[0],
                obstacle_params[1],
                Direction(obstacle_params[2]),
            ),
            obstacle_params[3],
        )
        obstacles.append(obstacle)
    return obstacles

def process_string_command(command: str):
    """Process string commands from RPI."""
    print(f"Processing string command: {command}")
    command_parts = command.split(',')

    try:
        # Handle NONE command with obstacle ID
        if len(command_parts) >= 2:
            obstacle_id = int(command_parts[1])
            commands = [
                StraightCommand(-10).convert_to_message(),
                ScanCommand(0, obstacle_id).convert_to_message(),
                StraightCommand(10).convert_to_message(),
            ]
            return commands
    except (IndexError, ValueError) as e:
        print(f"Error processing command: {e}")