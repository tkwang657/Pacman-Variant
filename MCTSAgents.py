from game import Agent
from math import sqrt, log
from MCTSTree import MCTSNode, EdgeFactory
from featureState import FeatureBasedState
from distanceCalculator import Distancer
#Actions: North East West South Stop
class MCTSAgent(Agent):
    """
    Returns a function that takes a position (an instance of State) and returns the best move
    after running MCTS for the specified time limit.
    """
    def __init__(self, index=0, num_train=100):
        super().__init__(index)
        self._distancer=None
        self._edgeFactory=EdgeFactory()
        self._currentgame=0
        self._num_train=num_train
        self._features=("direction_to_nearest_pellet", "Is_Ghost_Near_Me")
    
    def registerInitialState(self, gamestate): # inspects the starting state
        self._distancer=Distancer(layout=gamestate.data.layout)
        


    def getAction(self, gamestate):
        """Takes a GameState object"""
        rootfbs=FeatureBasedState(game_state=gamestate, features=self._features, distancer=self._distancer)
        root_tree_node=MCTSNode(state=rootfbs, edgefactory=self._edgeFactory)
        action=root_tree_node.explore(time_limit=0.4)
        print(action)
        return action
    


    

