from abc import ABC, abstractmethod
from pacman import GameState
import numpy as np

# class FeatureStateFactory():
#     def __init__(self):
#         self._features=None
#         self._visited={}
    
#     def setFeatures(self, features):
#         if self._features is None:
#             self._features=features
#         else:
#             raise Exception("Set Features called on factory already with objects")
#     def CreateState(self, gamestate, reward_function: str="WeightedScoringWithFoodCount"):
#         if self._features is None:
#             raise Exception("Cannot create Feature Based State without specifying features")
#         extractedFeaturesTuple=self._extractFeatures(gamestate)
#         if extractedFeaturesTuple in self._visited:
#             return self._visited[extractedFeaturesTuple]
#         else:
#             newState=FeatureBasedState(game_state=gamestate, reward_function=reward_function, features=extractedFeaturesTuple)
#             self._visited[extractedFeaturesTuple]=newState
#             return newState

#     def Count_states(self):
#         return len(self._visited)
    
#     def _extractFeatures(self, gamestate: GameState):
#         """Given a gamestate object, extract the features and returns them in a list of tuples"""
#         ghost_x, ghost_y=gamestate.getGhostPosition(agentIndex=1)
        # food_grid=np.array(gamestate.getFood().data, type=bool)
        # cur_x, cur_y=gamestate.getPacmanPosition()
        # data={"ghost_x": ghost_x,
        #       "ghost_y": ghost_y,
        #       "food_grid": food_grid, 
        #       "cur_x": cur_x, 
        #       "cur_y": cur_y, 
        #       "gamestate": gamestate}
        # featureTuple=[]
        # for feature in self._features:
        #     func_ptr=self._get_feature(feature_name=feature)
        #     featureTuple.append((feature, func_ptr(data=data)))
        # return tuple(featureTuple)

class FeatureBasedState():
    def __init__(self, game_state: GameState, features: tuple):
        self._raw_game_state=game_state
        self.features={}
        for feature in features:
            self.features[feature[0]]=feature[1]

    def getLegalActions(self, agentIndex=0):
        return self._raw_game_state.getLegalActions(agentIndex=agentIndex)
    
    def generateSuccessor(self, action="STOP"):
        """Returns raw game state given a certain action"""
        return self._raw_game_state.generatePacmanSuccessor(action=action)
    
    def isWin(self):
        return self._raw_game_state.isWin()
    
    def isLose(self):
        return self._raw_game_state.isLose()

    def is_terminal(self):
        if self._raw_game_state.isWin() or self._raw_game_state.isLose():
            return True
        else:
            return False
    def payoff(self):
        assert self.is_terminal(), "Payoff called on non-terminal state"
        return self.Heuristic_Evaluate()
    
    def Heuristic_Evaluate(self):
        pass

    

    def ZeroSumReward(self):
        if self._raw_game_state.isWin():
            return 1
        elif self._raw_game_state.isLose():
            return -1
        else:
            return 0
    
    def WeightedScoringWithFoodCountReward(self, w1=30, w2=10):
            
        
        successorGameScore = self._raw_game_state.getScore()

        # Actual calculations start here
        foodcount = self._raw_game_state.getNumFood()
        food_pos=np.argwhere(self._food_grid)
        x_distances=food_pos[:,0]-self._cur_x
        y_distances=food_pos[:,1]-self._cur_y
        sum_of_manhattan_distances = np.abs(x_distances)+np.abs(y_distances) #add the manhattan distances pairwise
        distanceFromClosestFood = 0 if (len(sum_of_manhattan_distances) == 0) else min(sum_of_manhattan_distances)

        finalScore = successorGameScore - (w1 * foodcount) - (w2 * distanceFromClosestFood) 
        return finalScore
    
    def __eq__(self, other: 'FeatureBasedState'):

        if len(self._features)==len(other._features):
            pass
        else:
            raise ValueError("Comparison between incompatible feature objects")
        for k,v in self._features.items():
            try:
                if self._features[k]==other._features[k]:
                    pass
                else:
                    return False
            except KeyError:
                raise ValueError("Comparison between incompatible feature objects")
        return True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        feature_strings = ", ".join(f"{key}={value}" for key, value in self._features.items())
        return f"FeatureBasedState({feature_strings})"
    
    


    