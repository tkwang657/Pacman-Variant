from game import Agent
from math import sqrt, log

#Actions: North East West South Stop
class MCTSAgent(Agent):
    def __init__(self):
        pass
    
    def _compute_UCT(self, wins, visits, N, c=sqrt(2)):
        return wins/visits + c * sqrt(log(N)/visits)


    

