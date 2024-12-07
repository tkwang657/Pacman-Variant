import time
from math import sqrt, log
import random
from featureEncoder import FeatureBasedState
class MCTSNode:
    def __init__(self, state: FeatureBasedState, parent=None):
        self.state = state         # The game state at this node
        self.parent = parent       # Parent node
        self.children = {}         # Map from actions to child nodes
        self.visits = 0            # Number of times this node has been visited
        self.reward = 0.0          # Total reward (payoff) for this node

    def is_fully_expanded(self):
        """Returns True if all children have been expanded."""
        return len(self.children) == len(self.state.getLegalActions())

    def is_terminal(self):
        return self.state.is_terminal()
    
    def selection(self): #Tree Policy is UCB
        """Returns the child with the highest UCB1 value to traverse, assuming that the node is fully expanded."""
        return max(self.children.values(), key=lambda child: self.compute_UCB(node=child))

    def compute_UCB(self, node, exploration_weight=sqrt(2)):
        if node.visits==0:
            return float('inf')
        else:
            if self.state.actor()==1:
                return  -node.reward / node.visits + exploration_weight * sqrt(log(self.visits) / node.visits)
            else:
                return node.reward / node.visits + exploration_weight * sqrt(log(self.visits) / node.visits)

    def expand(self):
        """Expands by adding arandom new child node for an unvisited action. Assumes that the node is expandable"""
        untried_actions = [
            action for action in self.state.getLegalActions() if action not in self.children
        ]
        action = random.choice(untried_actions)
        new_state = self.state.generateSuccessor(action)
        child_node = MCTSNode(new_state, parent=self)
        self.children[action] = child_node
        return child_node

    def rollout(self): #Default policy is random
        """Performs a random playout from this node's state to a terminal state and returns the payoff for player 0."""
        if self.state.is_terminal():
            return self.state.payoff()
        current_state = self.state
        while not current_state.is_terminal():
            possible_moves = current_state.getLegalActions()
            move = random.choice(possible_moves)
            current_state = current_state.generateSuccessor(move)
        return current_state.payoff()

    def backpropagate(self, reward):
        """Updates this node and its ancestors with the result of a rollout."""
        self.reward += reward if self.state.actor() == 0 else -reward  # Flip reward for player 1
        self.visits+=1
        parent=self
        while parent.parent:
            parent=parent.parent
            parent.visits+=1
            if parent.state.actor()==0:
                parent.reward+=reward
            else:
                parent.reward-=reward

        
    def explore(self, time_limit):
        """Runs MCTS from the root node for a given tiem limit in seconds"""
        end_time = time.process_time() + time_limit
        while time.process_time() < end_time:
            current=self
            #selection
            while not current.state.is_terminal() and current.is_fully_expanded():
                current=current.selection()
            #Expansion
            if not current.state.is_terminal() and not current.is_fully_expanded():
                current=current.expand()
            reward=current.rollout()
            current.backpropagate(reward=reward)
        best_action=max(current.children, key=lambda action: current.children[action].visits)
        return best_action




def mcts_policy(time_limit):
    """
    Returns a function that takes a position (an instance of State) and returns the best move
    after running MCTS for the specified time limit.
    """
    def policy_function(position):
        root = MCTSNode(position, parent=None)
        best_move = root.explore(time_limit)
        return best_move

    return policy_function
