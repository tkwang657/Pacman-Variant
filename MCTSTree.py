import time
from math import sqrt, log
import random
from featureState import FeatureBasedState
from pacman import GameState
import numpy as np
class NodeFactory:
    def __init__(self):
        self._features=None
        self._visited={}
    def setFeatures(self, features):
        if self._features is None:
            self._features=features
        else:
            raise Exception("Set Features called on factory already with objects")
    def CreateNode(self, gamestate, reward_function: str="WeightedScoringWithFoodCount"):
        if self._features is None:
            raise Exception("Cannot create Feature Based State without specifying features")
        extractedFeaturesTuple=self._extractFeatures(gamestate)
        if extractedFeaturesTuple in self._visited:
            return self._visited[extractedFeaturesTuple]
        else:
            newState=FeatureBasedState(game_state=gamestate, reward_function=reward_function, features=extractedFeaturesTuple)
            newNode=MCTSNode(state=newState, factory=self)
            self._visited[extractedFeaturesTuple]=newNode
            return newNode

    def CountNodes(self):
        return len(self._visited)
    


    #######
    #Calculate Features
    def _extractFeatures(self, gamestate: GameState):
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
            featureTuple.append((feature, func_ptr(data=data)))
        return tuple(featureTuple)


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


class MCTSEdge:
    def __init__(self, parent, child, action):
        self.parent=parent
        self.visits=0
        self.totalreward=0
        self.child=child
        self.action=action
    
    @property
    def q_value(self):
        if self.visits!=0:
            return self.totalreward/self.visits
        else:
            return float("inf")
    
    def update(self, reward):
        self.visits+=1
        self.totalreward+=reward
class MCTSNode:
    legal_actions=['North', 'East', 'South', 'West', 'Stop']
    def __init__(self, state: FeatureBasedState, factory: NodeFactory):
        self.state = state         #Feature based state
        self.edges = {}         # Map from actions to Edge object
        self.TotalVisits=0
        self._factory=factory


    def is_fully_expanded(self):
        """Returns True if all children have been expanded."""
        return len(self.edges) == 5

    def is_terminal(self):
        return self.state.is_terminal()
    
    def selection(self): #Tree Policy is UCB
        """Returns the action with the highest UCB1 value to traverse, assuming that root node is fully expanded."""
        return max(self.edges.keys(), key=lambda action: self.compute_UCB(edge=self.edges[action]))

    def compute_UCB(self, edge: MCTSEdge, exploration_weight=sqrt(2)):
        if edge.q_value== float("inf"):
            return float("inf")
        else:
            rtn= edge.q_value + exploration_weight*sqrt(log(self.TotalVisits)/edge.visits)
            return rtn
    def payoff(self):
        if not self.is_terminal():
            raise Exception("Payoff called on non-terminal node")
        else:
            return self.state.payoff

    def expand(self):
        """
        Returns: action
        Expands by adding a random new edge for an unvisited action. Assumes that the current node is expandable."""
        untried_actions = [action for action in MCTSNode.legal_actions if action not in self.edges]
        action = random.choice(untried_actions)
        raw_successor_state = self.state.generateSuccessor(action)
        child_node = self._factory.CreateNode(gamestate=raw_successor_state, parent=self)
        self.edges[action] = MCTSEdge(parent=self, child=child_node, action=action)
        return action

    def rollout(self, type="random"): #Default policy is random
        """Performs a random playout/heuristic score evaluation from this node and returns the payoff for player 0.
        Args: type='random' or 'Score'"""
        if type=="random":
            if self.is_terminal():
                return self.payoff()
            current_state = self.state._raw_game_state
            while not current_state.is_terminal():
                possible_moves = current_state.getLegalActions()
                move = random.choice(possible_moves)
                current_state = current_state.generateSuccessor(move)
            return current_state.payoff()
        elif type=="Score":
            return self.state.Heuristic_Evaluate()

    def backpropagate(self, reward, path):
        """Updates edges along the path taken to this node with the result of a rollout."""
        assert path[-1].child==self
        for edge in path[::-1]:
            edge.update(reward=reward)
            edge.parent.TotalVisits+=1
        

        
    def explore(self, time_limit):
        """Runs MCTS from the root node for a given tiem limit in seconds"""
        end_time = time.process_time() + time_limit
        while time.process_time() < end_time:
            current=self
            #Store a path to a node
            edgepath=[]
            #selection
            while not current.is_terminal() and current.is_fully_expanded():
                action=current.selection()
                edgepath.append(self.edges[action])
                current=self.edges[action].child
            #Expansion
            if not current.is_terminal() and not current.is_fully_expanded():
                action=current.expand()
                edgepath.append(self.edges[action])
                current=self.edges[action].child
            reward=current.rollout()
            current.backpropagate(reward, edgepath)
        best_action=max(current.edges.keys(), key=lambda action: current.edges[action].visits)
        return best_action




def mcts_policy(time_limit, features):
    """
    Returns a function that takes a position (an instance of State) and returns the best move
    after running MCTS for the specified time limit.
    """
    def policy_function(position):
        root = MCTSNode(position, parent=None)
        best_move = root.explore(time_limit)
        return best_move

    return policy_function
