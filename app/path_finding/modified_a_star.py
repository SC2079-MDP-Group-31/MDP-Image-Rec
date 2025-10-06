import heapq
import math
from typing import List, Tuple

from commands.command import Command
from commands.go_straight_command import StraightCommand
from commands.turn_command import TurnCommand
from grid.grid import Grid
from misc.direction import Direction
from misc.positioning import RobotPosition
from misc.type_of_turn import TypeOfTurn


class ModifiedAStar:
    def __init__(self, grid, brain, start: RobotPosition, end: RobotPosition, yolo):
        # We use a copy of the grid rather than use a reference to the exact grid.
        self.grid: Grid = grid.copy()
        self.brain = brain  # the Hamiltonian object
        self.start = start  # starting robot position (with direction)
        self.end = end  # target ending position (with direction)
        self.yolo = yolo

    def get_neighbours(self, pos: RobotPosition) -> List[Tuple[Tuple, RobotPosition, int, Command]]:
        """
        Get movement neighbours from this position.
        Note that all values in the Position object (x, y, direction) are all with respect to the grid!
        We also expect the return Positions to be with respect to the grid.
        """
        # We assume the robot will move by 10 when travelling straight, while moving a fixed x and y value when turning
        neighbours = []

        # Check travel straights.
        straight_dist = 10
        straight_commands = [
            StraightCommand(straight_dist),
            StraightCommand(-straight_dist),
        ]

        for command in straight_commands:
            # Check if doing this command does not bring us to any invalid position.
            after, p = self.check_valid_command(command, pos)
            if after:
                neighbours.append((after, p, straight_dist, command))

        # Check turns - ONLY MEDIUM turns now
        turn_penalty = 50 if not self.yolo else 0
        turn_commands = [
            TurnCommand(TypeOfTurn.MEDIUM, True, False, False),  # L MEDIUM turn, forward
            TurnCommand(TypeOfTurn.MEDIUM, True, False, True),  # L MEDIUM turn, reverse
            TurnCommand(TypeOfTurn.MEDIUM, False, True, False),  # R MEDIUM turn, forward
            TurnCommand(TypeOfTurn.MEDIUM, False, True, True),  # R MEDIUM turn, reverse
        ]

        for c in turn_commands:
            # Check if doing this command does not bring us to any invalid position.
            after, p = self.check_valid_command(c, pos)
            if after:
                neighbours.append((after, p, turn_penalty, c))

        return neighbours

    def check_valid_command(self, command: Command, p: RobotPosition):
        """
        Checks if a command will bring a point into any invalid position.

        If invalid, we return None for both the resulting grid location and the resulting position.
        """
        # Check specifically for validity of turn command. Robot should not exceed the grid or hit the obstacles
        p = p.copy()

        if isinstance(command, TurnCommand):
            p_c = p.copy()
            command.apply_on_pos(p_c)

            # make sure that the final position is a valid one
            if not (self.grid.check_valid_position(p_c, self.yolo)):
                return None, None

            # if positive means the new position is to the right, else to the left side
            diff_in_x = p_c.x - p.x
            # if positive means the new position is on top of old position, else otherwise
            diff_in_y = p_c.y - p.y

            # additional check for medium turn
            extraCheck = 0

            # displace to top right
            if diff_in_y > 0 and diff_in_x > 0:
                if p.direction == Direction.TOP or p.direction == Direction.BOTTOM:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p.copy()
                        temp.y += (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.x -= (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                else:  # rest of the directions
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.y -= (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p.copy()
                        temp.x += (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
            # displace to top left
            elif diff_in_x < 0 and diff_in_y > 0:
                if p.direction == Direction.TOP or p.direction == Direction.BOTTOM:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p.copy()
                        temp.y += (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.x += (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                else:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.y -= (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p.copy()
                        temp.x -= (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
            # displace to bottom left
            elif diff_in_x < 0 and diff_in_y < 0:
                if p.direction == Direction.LEFT or p.direction.RIGHT:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.y += (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p.copy()
                        temp.x -= (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                else:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p.copy()
                        temp.y -= (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.x += (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
            else:  # diff_in_x > 0 , diff_in_y < 0
                if p.direction == Direction.RIGHT or p.direction == Direction.LEFT:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.y += (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p.copy()
                        temp.x += (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                else:
                    for y in range(0, abs(diff_in_y // 10) + extraCheck):
                        temp = p.copy()
                        temp.y -= (y + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None
                    for x in range(0, abs(diff_in_x // 10) + extraCheck):
                        temp = p_c.copy()
                        temp.x -= (x + 1) * 10
                        if not (self.grid.check_valid_position(temp, self.yolo)):
                            return None, None

        command.apply_on_pos(p)

        if self.grid.check_valid_position(p) and (after := p.xy() + (p.get_dir(),)):
            return after, p
        return None, None

    def distance_heuristic(self, curr_pos: RobotPosition):
        """
        Measure the difference in distance between the provided position and the
        end position.
        """
        dx = abs(curr_pos.x - self.end.x)
        dy = abs(curr_pos.y - self.end.y)
        return math.sqrt(dx ** 2 + dy ** 2)

    def direction_heuristic(self, curr_pos: RobotPosition):
        """
        If not same direction as my target end position, incur penalty!
        """
        if self.end.direction == curr_pos.direction.value:
            return 0
        else:
            return 10

    def start_astar(self, flag):
        frontier = []  # Store frontier nodes to travel to as a priority queue.
        backtrack = dict()  # Store the sequence of grid cells being travelled.
        cost = dict()  # Store the cost to travel from start to a target grid cell.

        # We can check what the goal grid cell is
        goal_node: Tuple = self.end.xy()
        goal_node_with_dir: Tuple = goal_node + (self.end.direction,)

        # Add starting node set into the frontier.
        start_node: Tuple = self.start.xy()
        start_node_with_dir: Tuple = start_node + (self.start.direction,)

        offset = 0  # Used to tie-break

        # Extra time parameter to tie-break same priority.
        heapq.heappush(frontier, (0, offset, (start_node_with_dir, self.start)))
        cost[start_node_with_dir] = 0
        # Having None as the parent means this key is the starting node.
        backtrack[start_node_with_dir] = (None, None)  # Parent, Command

        while frontier:
            # Get the highest priority node.
            priority, _, (current_node, current_position) = heapq.heappop(frontier)

            # If the current node is our goal.
            if current_node == goal_node_with_dir:
                # Get the commands needed to get to destination.
                commands = self.extract_commands(backtrack, goal_node_with_dir, flag)
                return current_position, commands

            # Otherwise, we check through all possible locations that we can
            # travel to from this node.
            neighbours = self.get_neighbours(current_position)

            for new_node, new_pos, weight, c in neighbours:
                # weight here stands for cost of moving forward or turning
                revisit = 0

                if new_node in backtrack:
                    revisit = 10

                new_cost = cost.get(current_node) + weight + revisit

                if new_cost < cost.get(new_node, 100000):
                    offset += 1
                    priority = (
                            new_cost
                            + self.distance_heuristic(new_pos)
                            + self.direction_heuristic(new_pos)
                    )

                    heapq.heappush(frontier, (priority, offset, (new_node, new_pos)))
                    backtrack[new_node] = (current_node, c)
                    cost[new_node] = new_cost

        # If we are here, means that there was no path that we could find.
        # We return None to show that we cannot find a path.
        return None, []

    def extract_commands(self, backtrack, goal_node, flag):
        """
        Extract required commands to get to destination.
        """
        commands = []
        curr = goal_node

        while curr:
            curr, c = backtrack.get(curr, (None, None))
            if c:
                commands.append(c)

        commands.reverse()

        if flag:
            self.brain.commands.extend(commands)
            return None
        else:
            return commands

    def get_path_with_coordinates(self, flag):
        """
        Execute pathfinding and return commands with estimated coordinates after each movement.

        Args:
            flag: Processing flag for command extraction

        Returns:
            Tuple of (final_position, list_of_(command, estimated_position_after_command))
            or (None, []) if no path found
        """
        final_position, commands = self.start_astar(flag)

        if final_position is None or not commands:
            return None, []

        # Track coordinates after each command
        commands_with_coords = []
        current_pos = self.start.copy()

        for command in commands:
            # Apply command to get new position
            command.apply_on_pos(current_pos)
            # Store command with the position after execution
            commands_with_coords.append((command, current_pos.copy()))

        return final_position, commands_with_coords