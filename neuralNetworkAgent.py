import neat
import pickle
import layout
from pacman import ClassicGameRules
from ghostAgents import RandomGhost
from graphicsDisplay import PacmanGraphics
from utils import manhattanDistance
import time
import random

class NEATPacmanAgent:
    """
    A NEAT-controlled Pacman agent that uses a neural network to determine actions.
    """
    def __init__(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.previous_positions = []
        self.oscillation_count = 0
    
    def updateOscillationCount(self, current_position):
        """
        Detect and count oscillation based on position history.
        """
        self.previous_positions.append(current_position)
        if len(self.previous_positions) > 3:
            # Keep track of the last 3 positions
            self.previous_positions.pop(0)
            if len(set(self.previous_positions)) == 2:
                # Oscillation detected (alternating between 2 positions)
                self.oscillation_count += 1

    def getNextPosition(self, current_position, action):
        """
        Predicts Pacman's next position based on the current position and action.
        """
        x, y = current_position
        if action == "North":
            return (x, y + 1)
        elif action == "South":
            return (x, y - 1)
        elif action == "East":
            return (x + 1, y)
        elif action == "West":
            return (x - 1, y)
        return current_position

    def getAction(self, state):
        inputs = self.computeInputs(state)
        outputs = self.net.activate(inputs)
        
        # Define all possible actions without "Stop"
        actions = ["North", "South", "East", "West"]
        legal_actions = [action for action in actions if action in state.getLegalPacmanActions()]

        # Ensure there are legal actions to choose from
        if not legal_actions:
            return "North"  # Default to "North" if no legal actions are available

        # Randomization: Introduce a small probability of choosing a random action
        # Add epsilon randomness if more than 2 legal actions
        epsilon = 0.1  # Adjust this value to control randomness
        if len(legal_actions) > 2 and random.random() < epsilon:
            return random.choice(legal_actions)

        # Get Pacman's current position
        current_position = state.getPacmanPosition()

        # Track the positions that would lead back to recent locations
        avoid_actions = set()
        if len(self.previous_positions) > 1:
            for action in legal_actions:
                next_position = self.getNextPosition(current_position, action)
                if next_position in self.previous_positions[-2:]:
                    avoid_actions.add(action)

        # Filter legal actions to avoid oscillations
        non_oscillating_actions = [action for action in legal_actions if action not in avoid_actions]

        # Choose the best action
        if non_oscillating_actions:
            best_action_index = max(
                (i for i in range(len(outputs)) if actions[i] in non_oscillating_actions),
                key=lambda i: outputs[i],
                default=None,
            )
            best_action = actions[best_action_index] if best_action_index is not None else "North"
        else:
            # If all actions would lead to oscillation, pick the best legal action
            best_action_index = max(
                (i for i in range(len(outputs)) if actions[i] in legal_actions),
                key=lambda i: outputs[i],
                default=None,
            )
            best_action = actions[best_action_index] if best_action_index is not None else "North"

        # Update oscillation count and positions
        self.updateOscillationCount(current_position)
        return best_action


    def computeInputs(self, state):
        pacman_pos = state.getPacmanPosition()
        ghost_pos = state.getGhostPositions()
        food_positions = state.getFood().asList()

        # Nearest ghost distance
        nearest_ghost_distance = min(manhattanDistance(pacman_pos, ghost) for ghost in ghost_pos)
        # Nearest food distance
        nearest_food_distance = min(manhattanDistance(pacman_pos, food) for food in food_positions)

        # Compute local pellet density:
        # Define a search radius. Increase or decrease this as needed.
        radius = 5  
        local_pellet_count = 0

        for food in food_positions:
            if manhattanDistance(pacman_pos, food) <= radius:
                local_pellet_count += 1

        # Normalize local density
        # One simple normalization: divide by the maximum possible number of tiles in the radius.
        # For a radius r, a rough upper bound on tiles within r steps (in a Manhattan sense) could be:
        # number_of_tiles = sum of (1 + 2*i) for i=0 to r, which approximates the diamond shape of a Manhattan circle.
        # For simplicity, you could just assume a square: (2*radius+1)^2.
        # Or just keep it as a raw count if you prefer. Experiment to find what works best.
        max_possible_tiles = (2 * radius + 1) ** 2
        local_density = local_pellet_count / float(max_possible_tiles)

        return [
            nearest_ghost_distance / 100.0,  # normalize distances as before
            nearest_food_distance / 100.0,
            local_density  # local density instead of global pellet density
        ]


# def run_game(agent, visualize=True):
#     layout_name = 'smallClassic'
#     game_layout = layout.getLayout(layout_name)
#     if game_layout is None:
#         raise FileNotFoundError(f"The layout '{layout_name}' could not be found.")
#     rules = ClassicGameRules(timeout=30)
#     display = PacmanGraphics(zoom=1.0, frameTime=0.1) if visualize else None
#     ghost_agents = [RandomGhost(1)]
#     game = rules.newGame(game_layout, agent, ghost_agents, display, quiet=not visualize)
#     game.run()
#     return game.state.getScore()
def run_game(agents, visualize=True):
    """
    Runs multiple games simultaneously but displays only one game at a time.
    """
    layout_name = 'smallClassic'
    game_layout = layout.getLayout(layout_name)
    if game_layout is None:
        raise FileNotFoundError(f"The layout '{layout_name}' could not be found.")

    rules = ClassicGameRules(timeout=30)
    ghost_agents = [RandomGhost(1)]  # Add ghost agents

    # Create games for each agent
    games = [
        rules.newGame(game_layout, agent, ghost_agents, None, quiet=True)
        for agent in agents
    ]

    # Create a single display for visualization
    display = PacmanGraphics(zoom=1.0, frameTime=.1) if visualize else None

    # Initialize variables to track active game and finished games
    active_game_index = 0
    finished_games = set()
    scores = [-1] * len(games)  # Store scores for all games

    # Start the display with the first game
    if visualize:
        display.initialize(games[active_game_index].state.data)

    max_steps = 1000  # Limit steps to prevent infinite loops
    for step in range(max_steps):
        # time.sleep(0.1)
        # Check if all games are finished
        if all(game.state.isWin() or game.state.isLose() for game in games):
            break

        # Update active game
        active_game = games[active_game_index]

        # Run one step for all games
        for i, game in enumerate(games):
            if i in finished_games:
                continue  # Skip finished games

            if not (game.state.isWin() or game.state.isLose()):
                # Get Pacman's action
                pacman_action = game.agents[0].getAction(game.state)
                game.state = game.state.generateSuccessor(0, pacman_action)

                # Get ghost's action
                if not (game.state.isWin() or game.state.isLose()):
                    ghost_action = game.agents[1].getAction(game.state)
                    game.state = game.state.generateSuccessor(1, ghost_action)

            # Check if the game has ended
            if game.state.isWin() or game.state.isLose():
                # print(f"Game {i} finished. Score: {game.state.getScore()}")
                scores[i] = game.state.getScore()
                finished_games.add(i)

        # Update display for the active game
        if visualize:
            display.update(active_game.state.data)

            # If the active game has ended, switch to another ongoing game
            if active_game_index in finished_games:
                ongoing_games = [i for i in range(len(games)) if i not in finished_games]
                if ongoing_games:
                    active_game_index = ongoing_games[0]
                    display.initialize(games[active_game_index].state.data)

    return scores

# def eval_genomes(genomes, config):
#     """
#     Fitness evaluation function for NEAT. Each genome is tested in a game.
#     """
#     for genome_id, genome in genomes:
#         agent = NEATPacmanAgent(genome, config)
#         score = run_game(agent, visualize=True)  # Set visualize=True for debugging
#         genome.fitness = score  # Fitness is the game score
def eval_genomes(genomes, config):
    """
    Evaluates all genomes in a single generation by running them simultaneously on the same board.
    """
    # Create a NEATPacmanAgent instance for each genome
    agents = [NEATPacmanAgent(genome, config) for _, genome in genomes]

    # Run the games for all agents in the generation simultaneously
    scores = run_game(agents, visualize=True)  # Pass the list of agents

    # Assign fitness scores to each genome based on the game scores
    for (genome_id, genome), score, agent in zip(genomes, scores, agents):
        oscillation_penalty = agent.oscillation_count * 100
        genome.fitness = score - oscillation_penalty

def run_winner(winner_genome, config, layout_name='smallClassic', visualize=True):
    """
    Runs a game using the winner genome from NEAT training with DirectionalGhosts.
    """
    from ghostAgents import DirectionalGhost  # Import the DirectionalGhost class

    # Create a NEAT-controlled Pacman agent
    winner_agent = NEATPacmanAgent(winner_genome, config)

    # Load the layout
    game_layout = layout.getLayout(layout_name)
    if game_layout is None:
        raise FileNotFoundError(f"The layout '{layout_name}' could not be found.")

    # Set up the game
    rules = ClassicGameRules(timeout=30)

    # Use DirectionalGhost as the ghost agent
    ghost_agents = [DirectionalGhost(i + 1) for i in range(game_layout.getNumGhosts())]

    # Set up the display
    display = PacmanGraphics(zoom=1.0, frameTime=0.1) if visualize else None

    # Create and run the game
    game = rules.newGame(game_layout, winner_agent, ghost_agents, display, quiet=not visualize)
    game.run()

    # Return the final score
    return game.state.getScore()

# def run_winner(winner_genome, config, layout_name='smallClassic', visualize=True):
#     """
#     Runs a game using the winner genome from NEAT training.
#     """
#     # Create a NEAT-controlled Pacman agent
#     winner_agent = NEATPacmanAgent(winner_genome, config)

#     # Load the layout
#     game_layout = layout.getLayout(layout_name)
#     if game_layout is None:
#         raise FileNotFoundError(f"The layout '{layout_name}' could not be found.")

#     # Set up the game
#     rules = ClassicGameRules(timeout=30)
#     ghost_agents = [RandomGhost(1)]  # Add a ghost to interact with Pacman
#     display = PacmanGraphics(zoom=1.0, frameTime=0.1) if visualize else None

#     # Create and run the game
#     game = rules.newGame(game_layout, winner_agent, ghost_agents, display, quiet=not visualize)
#     game.run()

#     # Return the final score
#     return game.state.getScore()

import statistics  # For calculating variance
import textDisplay

def test_winner(winner_genome, config, layout_name='smallClassic', time_limit=4*60*60, visualize=False):
    """
    Tests the best genome for a specified time limit, tracking mean and variance of scores.
    """
    from ghostAgents import DirectionalGhost  # Use DirectionalGhost for testing

    # Create a NEAT-controlled Pacman agent
    winner_agent = NEATPacmanAgent(winner_genome, config)

    # Load the layout
    game_layout = layout.getLayout(layout_name)
    if game_layout is None:
        raise FileNotFoundError(f"The layout '{layout_name}' could not be found.")

    # Set up the game
    rules = ClassicGameRules(timeout=30)
    ghost_agents = [DirectionalGhost(i + 1) for i in range(game_layout.getNumGhosts())]

    # Use NullGraphics if visualize is False
    display = PacmanGraphics(zoom=1.0, frameTime=0.1) if visualize else textDisplay.NullGraphics()

    # Track scores
    scores = []
    start_time = time.time()

    while time.time() - start_time < time_limit:
        # Run a game
        game = rules.newGame(game_layout, winner_agent, ghost_agents, display, quiet=not visualize)
        game.run()
        scores.append(game.state.getScore())

    # Calculate mean and variance
    mean_score = sum(scores) / len(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0

    print(f"\nTesting Results:")
    print(f"Total Games Played: {len(scores)}")
    print(f"Mean Score: {mean_score}")
    print(f"Variance: {variance}")

    return mean_score, variance



import argparse  # Add this import for command-line argument parsing


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run NEAT Pacman agent.")
    parser.add_argument("-best", action="store_true", help="Run the saved best genome from 'winner.pkl'")
    parser.add_argument("-test", action="store_true", help="Test the saved best genome for 4 hours")
    args = parser.parse_args()

    if args.best or args.test:
        # Run the best genome saved in winner.pkl
        try:
            with open("winner.pkl", "rb") as f:
                winner = pickle.load(f)
            print("\nLoaded best genome:\n", winner)

            # Load NEAT configuration
            config_path = "neat-config.ini"  # Path to the config file
            config = neat.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
            )

            if args.best:
                # Run the game with the loaded winner
                final_score = run_winner(winner, config, visualize=True)
                print(f"Winner's final score: {final_score}")

            elif args.test:
                # Test the winner for 4 hours
                test_winner(winner, config, time_limit=4*60*60, visualize=False)

        except FileNotFoundError:
            print("Error: 'winner.pkl' not found. Train a model first to save the best genome.")
    else:
        # Load NEAT configuration
        config_path = "neat-config.ini"  # Path to the config file
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
        )

        # Create the NEAT population
        population = neat.Population(config)

        # Add reporters to show progress in the console
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        # Run NEAT for 100 generations
        winner = population.run(eval_genomes, n=100)

        # Save the best genome
        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)

        print("\nBest genome:\n", winner)

        # Run the game with the winner
        final_score = run_winner(winner, config, visualize=True)
        print(f"Winner's final score: {final_score}")
        # Load NEAT configuration
        # config_path = "neat-config.ini"  # Path to the config file
        # config = neat.Config(
        #     neat.DefaultGenome, neat.DefaultReproduction,
        #     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
        # )

        # # Create the NEAT population
        # population = neat.Population(config)

        # # Add reporters to show progress in the console
        # population.add_reporter(neat.StdOutReporter(True))
        # stats = neat.StatisticsReporter()
        # population.add_reporter(stats)

        # # Time-based training logic
        # time_limit = 6 * 60 * 60  # 6 hours in seconds
        # start_time = time.time()
        # best_genome = None

        # # Run NEAT until the time limit is reached
        # generation = 0
        # while time.time() - start_time < time_limit:
        #     generation += 1
        #     print(f"Starting generation {generation}...")
        #     best_genome = population.run(eval_genomes, n=1)  # Run one generation at a time

        # # Save the best genome after training ends
        # with open("winner.pkl", "wb") as f:
        #     pickle.dump(best_genome, f)

        # print("\nBest genome:\n", best_genome)

        # # Run the game with the winner
        # final_score = run_winner(best_genome, config, visualize=True)
        # print(f"Winner's final score: {final_score}")