from distanceCalculator import Distancer
from pacman import GameState
import numpy as np
import random
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
    def __init__(self, game_state: GameState, features: tuple, distancer: Distancer):
        self.raw_game_state=game_state
        self.features=features
        self.distancer=distancer
        #Useful Intermmediate data
        ghost_x, ghost_y=self.raw_game_state.getGhostPosition(agentIndex=1)
        food_grid=np.argwhere(np.array(self.raw_game_state.getFood().data, dtype=bool))
        cur_x, cur_y=self.raw_game_state.getPacmanPosition()
        self.data={"ghost_x": ghost_x,
              "ghost_y": ghost_y,
              "food_grid": food_grid, 
              "cur_x": cur_x, 
              "cur_y": cur_y}
        self._nearest_food=self._distance_and_direction_to_nearest_pellet(self.data)
        #print("INIT: ", self._nearest_food)
    def getLegalActions(self, agentIndex=0):
        return self.raw_game_state.getLegalActions(agentIndex=agentIndex)
    
    def generateSuccessor(self, action="STOP"):
        """Returns raw game state given a certain action"""
        raw= self.raw_game_state.generatePacmanSuccessor(action=action)
        return FeatureBasedState(game_state=raw, features=self.features, distancer=self.distancer)
    
    def isWin(self):
        return self.raw_game_state.isWin()
    
    def isLose(self):
        return self.raw_game_state.isLose()

    def is_terminal(self):
        if self.raw_game_state.isWin() or self.raw_game_state.isLose():
            return True
        else:
            return False
    def payoff(self):
        assert self.is_terminal(), "Payoff called on non-terminal state"
        return self.Heuristic_Evaluate()
    
    def Heuristic_Evaluate(self):
        if self.is_terminal():
            if self.isWin():
                return 500
            else:
                return -500
        else:
            #return 1/self._distance_to_nearest_pellet(self.data) + 1-len(self.data["food_grid"])/100
            return self.raw_game_state.getScore()/100-self._distance_to_nearest_pellet(self.data)+10/len(self.data["food_grid"])+ random.randint(0,2)
        

    

    #######
    #Calculate Features
    def extractFeatures(self):
        featureTuple=[]
        for feature in self.features:
            func_ptr=self._get_feature(feature_name=feature)
            featureTuple.append((feature, func_ptr(data=self.data)))
        return tuple(featureTuple)


    def _Weighted_Food_Distances(self, data: dict):
        # Array of (x, y) positions where food is located
        food_pos=data["food_grid"]
        #Calculate center of food
        x_center=np.mean(food_pos[:, 0]-data["cur_x"])
        y_center=np.mean(food_pos[:,1]-data["cur_y"])
        #average_manhattan_distance
        return x_center, y_center

    def _Food_In_Three_Blocks(self, data: dict):
        food_pos=data["food_grid"]
        for food in food_pos:
            if self.distancer.getDistance(food, pos2=(data["cur_x"], data["cur_y"])):
                return True
            else:
                return False
    
    def _distance_and_direction_to_nearest_pellet(self, data: dict):
        if self.is_terminal():
            return None
        food_pos=data["food_grid"]
        foods=[]
        for food in food_pos:
            foods.append((food, self.distancer.getDistance(food, pos2=(data["cur_x"], data["cur_y"]))))
        itm=min(foods, key=lambda x: x[1])
        directions = {"North": (0, 1),
                   "South": (0, -1),
                   "East":  (1, 0),
                   "West":  (-1, 0)}
        for action in self.raw_game_state.getLegalActions():
            if action=="Stop":
                continue
            else:
                vector=directions[action]
                new_pos=(self.data["cur_x"]+vector[0], self.data["cur_y"]+vector[1])
                #print("iterating: ", (self.data["cur_x"], self.data["cur_y"]), action, new_pos, itm, self.distancer.getDistance(itm[0], pos2=new_pos))
                if self.distancer.getDistance(itm[0], pos2=new_pos)<itm[1]:
                    return itm, action
                else:
                    pass
        assert self.is_terminal()==False, print("blab")
        #print("testing", [a for a in self.raw_game_state.getLegalActions() if a!="Stop"])
        return itm, random.choice([a for a in self.raw_game_state.getLegalActions() if a!="Stop"])
    
    def _distance_to_nearest_pellet(self, data: dict):
        return self._nearest_food[0][1]
    def _direction_to_nearest_pellet(self, data: dict):
        return self._nearest_food[1]
    def _Is_Ghost_Near_Me(self, data: dict):
        return self._Is_Ghost_In_One_Block(data) or self._Is_Ghost_In_Two_Blocks(data)
    
    # def getMoveToClosestFood(self, data: dict):
    #     moves=data["gamestate"]
    #     problem = searchAgents.AnyFoodSearchProblem(self.rawGameState)
    #     sequenceOfActions = search.aStarSearch(problem)
    #     return sequenceOfActions[0]

    def _Is_Ghost_In_One_Block(self, data: dict):
        # Check if ghost is in a 2 block radius by manhattan distance
        manhattan = np.abs(data["cur_x"]-data["ghost_x"])+np.abs(data["cur_y"]-data["ghost_y"])
        return manhattan<=1
    
    def _Is_Ghost_In_Two_Blocks(self, data: dict):
        #IDS search_algorithm
        ghost_pos=(data["ghost_x"], data["ghost_y"])
        pac_pos=(data["cur_x"], data["cur_y"])
        return self.distancer.getDistance(pos1=ghost_pos, pos2=pac_pos)<=2

#Notes: Ghosts are index 1
        ######
    #Helper functions
    def _get_feature(self, feature_name: str):
        method=getattr(self, f"_{feature_name}")
        if method:
            return method
        else:
            raise ValueError(f"Feature '{feature_name}' not found.")
    

    # def __str__(self):
    #     return repr(self)

    # def __repr__(self):
    #     feature_strings = ", ".join(f"{key}={value}" for key, value in self.features.items())
    #     return f"FeatureBasedState({feature_strings})"
    
    
    


    