from utils import manhattanDistance
from collections import deque
from game import Directions, Actions
from game import Agent
import random
import game
import utils
class GreedyFoodAgent(Agent):
    """
    An agent that moves Pacman greedily toward the nearest food pellet
    based on the shortest path (not just Manhattan distance).
    """

    def getAction(self, state):
        pacman_position = state.getPacmanPosition()
        food_grid = state.getFood()
        legal_actions = state.getLegalActions(0)

        # Get the list of all food positions
        food_positions = food_grid.asList()

        if not food_positions:
            # No food left; stop
            return Directions.STOP

        # Find the closest food using BFS
        closest_food, path_to_food = self.findClosestFood(state, pacman_position, food_positions)

        # Return the first action in the path
        if path_to_food:
            return path_to_food[0]
        else:
            return Directions.STOP

    def findClosestFood(self, state, start, food_positions):
        """
        Perform BFS to find the closest food and the path to it.
        """
        walls = state.getWalls()
        queue = deque([(start, [])])  # (current_position, path)
        visited = set()

        while queue:
            position, path = queue.popleft()

            if position in visited:
                continue
            visited.add(position)

            # If we found food, return the position and path
            if position in food_positions:
                return position, path

            # Add neighbors to the queue
            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(direction)
                next_position = (int(position[0] + dx), int(position[1] + dy))

                if not walls[next_position[0]][next_position[1]] and next_position not in visited:
                    queue.append((next_position, path + [direction]))

        return None, []  # No food found


def scoreEvaluation(state):
    return state.getScore()
