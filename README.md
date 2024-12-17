Make sure the packages in requirements.txt are installed before running

To run pacman interactively with keyboard:
`python3 pacman.py --layout <nameofmap>`
For example:
`python3 pacman.py --layout smallClassic`

**Description of Game**
Welcome to Pacman! The goal of the game is to have the yellow Pac-man eat all of the pellets before any of the ghosts touch you! For simplicity, we have removed
fruit, all but one ghost, and created a smaller custom board to run the game on.

**MCTS Agent** (Alex Kwang)
To test the MCTS Agent, run:
python3 pacman.py --layout <nameofmap> -p MCTSAgent -g <GhostType> --quietTextGraphics --numGames <numGames> --maxDuration <maxDuration> -a timelimit=timelimit
1. layout: bigClassic, mediumClassic, smallClassic, smallClassicLittleFood
2. GhostType: RandomGhost or DirectionalGhost (Note the DirectionalGhost is a greedyghost that chases Pacman)
4. the --quietTextGraphics flag is passed in to avoid any graphics (The Zoo does not support tkinter so run with quietTextGraphics)
5. If both maxDuration and numgames are passed in, the code will terminate whichever limit is hit first. maxDuration is amount of time in minutes.
5. -a passes in a time limit in seconds for the MCTS Agent for each action. The default is 0.05 seconds
Example: 
python3 pacman.py --layout smallClassic -p MCTSAgent -g DirectionalGhost --quietTextGraphics --numGames 20 --maxDuration 5 -a timelimit=0.05
will run the game for either 6 minutes or 20 games and output stats whichever is hit first, where our MCTS agent has 0.05 seconds per move.

**Neural Network Agent** (Thomas Chung)
To train the Neural Network, run:
`python3 neuralNetworkAgent.py`
This will train the NN for whatever number of generations specified by the code (right now n=100), but if you switch the else statement of the __main__ function
with what is commented out, it will run for a certain amount of time (default = 4 hours).

To run the best Neural Network you have saved (a default is already loaded in in winner.pkl) once, run
`python3 neuralNetworkAgent.py -best`

To test the Neural Network agent and return the mean & variance over a disclosed amount of time (right now 30 seconds), run
`python3 neuralNetworkAgent.py -test` --> results will print out in 30 seconds

**Results**
For the Neural Network, the agent trained for 6 hours and ran through 2242 generations. The factors used were distance to the nearest (only) ghost, the distance 
to the nearest food pellet, and the local pellet density in a 5 tile radius (manhattan distance). We then tested the agent for 4 hours on both Greedy and Random agents.
Greedy Agent:
  Total Games Played: 2876636
  Mean Score: -320.586781574033
  Variance: 1725.755372139545
Random Agent:
  Total Games Played: 143316
  Mean Score: -296.09742108347984
  Variance: 10467.106219215242

