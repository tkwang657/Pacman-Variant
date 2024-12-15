from abc import ABC, abstractmethod
from pacman import GameState
import numpy as np

class FeatureStateFactory():
    def __init__(self, features: list[str]):
        self._features=features
        self._visited={}
    
    def CreateState(self, gamestate, reward_function: str="WeightedScoringWithFoodCount"):
        extractedFeaturesTuple=self._extractFeatures(gamestate)
        if extractedFeaturesTuple in self._visited:
            return self._visited[extractedFeaturesTuple]
        else:
            newState=FeatureBasedState(game_state=gamestate, reward_function=reward_function, features=extractedFeaturesTuple)
            self._visited[extractedFeaturesTuple]=newState
            return newState

    def Count_states(self):
        return len(self._visited)
    
    def _extractFeatures(self, gamestate: GameState):
        """Given a gamestate object, extract the features and returns them in a list of tuples"""
        ghost_x, ghost_y=gamestate.getGhostPosition(agentIndex=1)
        food_grid=np.array(gamestate.getFood().data, type=bool)
        cur_x, cur_y=gamestate.getPacmanPosition()
        data={"ghost_x": ghost_x,
              "ghost_y": ghost_y,
              "food_grid": food_grid, 
              "cur_x": cur_x, 
              "cur_y": cur_y, 
              "gamestate": gamestate}
        featureTuple=[]
        for feature in self._features:
            func_ptr=self._get_feature(feature_name=feature)
            featureTuple.append((feature, func_ptr(gamestate)))
        return tuple(featureTuple)


    #######
    #Calculate Features
    def _Weighted_Food_Distances(self, data: dict):
        # Array of (x, y) positions where food is located
        food_pos=np.argwhere(data["food_grid"])
        assert data["gamestate"].getNumFood()==len(food_pos)
        #Calculate center of food
        x_center=np.mean(food_pos[:, 0]-data["cur_x"])
        y_center=np.mean(food_pos[:,1]-data["cur_y"])
        #average_manhattan_distance
        return x_center, y_center

    def _Food_In_Three_Blocks(self, data: dict):
        pass

    def _nearest_pellet(self, data: dict):
        #Nearest food pellet by manhattan distance, if multiple, return multiple
        food_pos=np.argwhere(data["food_grid"])
        distances = np.abs(food_pos[:, 0] - data["cur_x"]) + np.abs(food_pos[:, 1] - data["cur_y"])
        min_distance = np.min(distances)
        nearest_pellets = food_pos[distances == min_distance]
        return nearest_pellets
        
    def _Is_Ghost_In_One_Block(self, data: dict):
        # Check if ghost is in a 2 block radius by manhattan distance
        manhattan = np.abs(data["cur_x"]-data["ghost_x"])+np.abs(data["cur_y"]-data["ghost_y"])
        return manhattan<=1
    
    def _Is_Ghost_In_Two_Blocks(self, data: dict):
        #IDS search_algorithm
        pass

#Notes: Ghosts are index 1
        ######
    #Helper functions
    def _get_feature(self, feature_name: str):
        method=getattr(self, f"_{feature_name}")
        if method:
            return method
        else:
            raise ValueError(f"Feature '{feature_name}' not found.")
class FeatureBasedState():
    def __init__(self, game_state: GameState, reward_function, features: tuple):
        self._raw_game_state=game_state
        self._reward_function=getattr(self, reward_function)
        if not callable(self._reward_function):
            raise ValueError(f"Reward function '{reward_function}' is not callable or does not exist.")
        self.features={}
        for feature in features:
            self.features[feature[0]]=feature[1]

    def getLegalActions(self, agentIndex=0):
        return self._raw_game_state.getLegalActions(agentIndex=agentIndex)
    
    def generateSuccessor(self, action="STOP"):
        return FeatureBasedState(game_state=self._raw_game_state.generatePacmanSuccessor(action=action), features=list(self._features.keys()))
    
    def is_terminal(self):
        if self._raw_game_state.isWin() or self._raw_game_state.isLose():
            return True
        else:
            return False
    def payoff(self):
        func=getattr(self, f"{self._reward_function}Reward", None)
        return func()

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
    
    


    