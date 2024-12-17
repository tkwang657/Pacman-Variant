
from collections import deque
from game import Directions, Actions
from game import Agent
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

from game import Agent
from MCTSTree import MCTSNode, EdgeFactory
from featureState import FeatureBasedState
from distanceCalculator import Distancer
#Actions: North East West South Stop
class MCTSAgent(Agent):
    """
    Returns a function that takes a position (an instance of State) and returns the best move
    after running MCTS for the specified time limit.
    """
    def __init__(self, index=0, timelimit=0.05):
        super().__init__(index)
        self._distancer=None
        self._edgeFactory=EdgeFactory()
        self._features=("direction_to_nearest_pellet", "Is_Ghost_Near_Me")
        self._timelimit=float(timelimit)
    
    def registerInitialState(self, gamestate): # inspects the starting state
        self._distancer=Distancer(layout=gamestate.data.layout)
        


    def getAction(self, gamestate):
        """Takes a GameState object"""
        rootfbs=FeatureBasedState(game_state=gamestate, features=self._features, distancer=self._distancer)
        root_tree_node=MCTSNode(state=rootfbs, edgefactory=self._edgeFactory)
        action=root_tree_node.explore(time_limit=self._timelimit)
        #print(action)
        return action
    
    def reset(self):
        self._edgeFactory=EdgeFactory()

    


def scoreEvaluation(state):
    return state.getScore()
