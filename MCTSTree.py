import time
from math import sqrt, log
import random
from featureState import FeatureBasedState
class EdgeFactory:
    def __init__(self):
        self._visited={}
    def CreateEdge(self, feature_tuple, action):
        key=(feature_tuple, action)
        if key in self._visited:
            return self._visited[key]
        else:
            new_edge=MCTSEdge(parent=feature_tuple, action=action)
            self._visited[key]=new_edge
            return new_edge
    


class MCTSEdge:
    def __init__(self, parent: tuple, action):
        """Pass in the feature tuple as the parent"""
        self.parent=parent
        self.visits=0
        self.totalreward=0
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
    def __init__(self, state: FeatureBasedState, edgefactory: EdgeFactory):
        self.state = state         #Feature based game statestate
        self._edgefactory=edgefactory
        self.actions={}
    
    def selection(self, epsilon=0.5): #Tree Policy is UCB
        ###
        # u=random.random()
        # if u<=exploitation and not self.state.features["Is_Ghost_Near_Me"]:
        #     return

        """Returns the action with the highest UCB1 value to traverse, assuming that root node is fully expanded."""
        u=random.random()
        if u<=epsilon and not self.state._Is_Ghost_Near_Me(self.state.data):
            return self.state._nearest_food[1]
        else:
            return max(self.state.getLegalActions(), key=lambda action: self.compute_UCB(action=action))

    def is_Fully_Expanded(self):
        return len(self.actions)==len(self.state.getLegalActions())

    def compute_UCB(self, action, exploration_weight=sqrt(2)):
        edge=self.action_to_edge(action=action)
        if edge.q_value== float("inf") or edge.visits==0:
            return float("inf")
        else:
            rtn= edge.q_value + exploration_weight*sqrt(log(self.TotalVisits)/edge.visits)
            return rtn

    def payoff(self):
        if not self.state.is_terminal():
            raise Exception("Payoff called on non-terminal node")
        else:
            return self.state.payoff

    def expand(self):
        """
        Returns: action
        Expands by adding a random new edge for an unvisited action. Assumes that the current node is expandable."""
        untried_actions = [action for action in self.state.getLegalActions() if action not in self.actions.keys()]
        action = random.choice(untried_actions)
        new_state=self.state.generateSuccessor(action=action)
        self.actions[action]=MCTSNode(state=new_state, edgefactory=self._edgefactory)
        self._edgefactory.CreateEdge(feature_tuple=self.state.extractFeatures(), action=action)
        return action

    def rollout(self, type="Score"): #Default policy is random
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
        else:
            raise ValueError(f"Invalid rollout type {type}")

    def backpropagate(self, reward, path):
        """Updates edges along the path taken to this node with the result of a rollout."""
        for edge in path[::-1]:
            edge.update(reward=reward)
        

        
    def explore(self, time_limit):
        """Runs MCTS from the root node for a given tiem limit in seconds"""
        end_time = time.process_time() + time_limit
        i=0
        while time.process_time() < end_time:
            current=self
            #Store a path to a node
            edgepath=[]
            #selection
            while not current.state.is_terminal() and current.is_Fully_Expanded():
                action=current.selection()
                edgepath.append(self.action_to_edge(action=action))
                current=MCTSNode(state=self.state.generateSuccessor(action=action), edgefactory=self._edgefactory)
            #Expansion
            if not current.state.is_terminal() and not current.is_Fully_Expanded():
                action=current.expand()
                edgepath.append(self.action_to_edge(action=action))
                current=current.actions[action]
            reward=current.rollout()
            #print("are we stuck here")
            current.backpropagate(reward, edgepath)
            i+=1
        best_action=self.selection()
        # print("EdgeData:")
        # ft=self.state.extractFeatures()
        # for k, v in self._edgefactory._visited.items():
        #     if k[0] == ft:
        #         print(f"{k}: {v.visits}")
        # print(self.state._nearest_food)
        return best_action
    @property
    def TotalVisits(self):
        total=0
        for action in self.actions:
            total+=self.action_to_edge(action=action).visits
        return total
            
    def action_to_edge(self, action):
        return self._edgefactory.CreateEdge(feature_tuple=self.state.extractFeatures(), action=action)





