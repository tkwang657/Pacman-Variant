from abc import ABC, abstractmethod
from pacman import GameState
import numpy as np
class FeatureBasedState():
    def __init__(self, game_state: GameState, features: list[str], reward_function: str="WeightedScoringWithFoodCount"):

        self._raw_game_state=game_state
        self._reward_function=reward_function
        #Useful intermediate information
        self._ghost_location=self._raw_game_state.getGhostPosition(agentIndex=1)
        self._food_grid=np.array(self._raw_game_state.getFood().data, dtype=bool) #Boolean grid of whether there is food at a certain spot on the grid. Each point is accessed via data[x][y]
        self._cur_x, self._cur_y=self._raw_game_state.getPacmanPosition() #(x, y) tuple
        #Build the features
        self._features={}
        for feature in features:
            func_ptr=self._get_feature(feature_name=feature)
            self._features[feature]=func_ptr()
    
    def __hash__(self):

        pass

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

    #######
    #Calculate Features
    def _Weighted_Food_Distances(self):
        # Array of (x, y) positions where food is located
        food_pos=np.argwhere(self._food_grid)
        assert self._raw_game_state.getNumFood()==len(food_pos)
        #Calculate center of food
        x_center=np.mean(food_pos[:, 0]-self._cur_x)
        y_center=np.mean(food_pos[:,1]-self._cur_y)
        #average_manhattan_distance
        return x_center, y_center

    def _Food_In_Three_Blocks(self):
        pass

    def _nearest_pellet(self):
        #Nearest food pellet by manhattan distance, if multiple, return multiple
        food_pos=np.argwhere(self._food_grid)
        distances = np.abs(food_pos[:, 0] - self._cur_x) + np.abs(food_pos[:, 1] - self._cur_y)
        min_distance = np.min(distances)
        nearest_pellets = food_pos[distances == min_distance]
        return nearest_pellets
        
    def _Is_Ghost_Near_Next_Move(self, move):
        if move == "North":
            next_pos = (self._cur_x, self._cur_y + 1)
        elif move == "South":
            next_pos = (self._cur_x, self._cur_y - 1)
        elif move == "East":
            next_pos = (self._cur_x + 1, self._cur_y)
        elif move == "West":
            next_pos = (self._cur_x - 1, self._cur_y)
        else:
            raise Exception("Invalid move " + str(self.moveToClosestFood))
        cpx, cpy = next_pos
        # Check if ghost is in a 2 block radius by manhattan distance
        manhattan = np.abs(cpx-next_pos[0])+np.abs(cpy-next_pos[1])
        return manhattan<=2
#Notes: Ghosts are index 1
    
    ######
    #Helper functions
    def _get_feature(self, feature_name: str):
        method=getattr(self, f"_{feature_name}")
        if method:
            return method
        else:
            raise ValueError(f"Feature '{feature_name}' not found.")

    