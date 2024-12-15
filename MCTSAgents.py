# from game import Agent
# from math import sqrt, log
# from MCTSTree import MCTSNode
# from featureEncoder import FeatureBasedState
# #Actions: North East West South Stop
# class MCTSAgent(Agent):
    
<<<<<<< Updated upstream
    def getAction(self, gamestate):
        """Takes a GameState object"""
        fbgs=FeatureBasedState(game_state=gamestate, features=["Weighted_Food_Distances", "Food_In_Three_Blocks", "Nearest_Food", "Is_Ghost_Near_Next_Move"])         # The game state at this node
        root_node=MCTSNode(state=fbgs, parent=None)
        # best_move = root_node.explore(time_limit=)
=======
#     def getAction(self, gamestate):
#         """Takes a GameState object"""
#         fbgs=FeatureBasedState(game_state=gamestate, features=["Weighted_Food_Distances", "Food_In_Three_Blocks", "Nearest_Food", "Is_Ghost_Near_Next_Move"])         # The game state at this node
#         root_node=MCTSNode(state=fbgs, parent=None)
#         best_move = root_node.explore(time_limit=None)
>>>>>>> Stashed changes
    
#     def _compute_UCT(self, wins, visits, N, c=sqrt(2)):
#         return wins/visits + c * sqrt(log(N)/visits)


    

