Make sure the packages in requirements.txt are installed before running

To run pacman interactively with keyboard:
`python3 pacman.py --layout <nameofmap>`
For example:
`python3 pacman.py --layout smallClassic`

**Description of Game**
Welcome to Pacman! The goal of the game is to have the yellow Pac-man eat all of the pellets before any of the ghosts touch you! For simplicity, we have removed all pellts and fruits, cut down the # of ghosts to 1, and created a custom board to run the game on.

Research Question: "How do features contribute more effectively: as state encoders for search efficiency, or as inputs to heuristics for state valuation?"

**MCTS Agent** (Alex Kwang)

To test the MCTS Agent, run:
python3 pacman.py --layout <nameofmap> -p MCTSAgent -g <GhostType> --quietTextGraphics --numGames <numGames> --maxDuration <maxDuration> -a timelimit=timelimit
1. layout: bigClassic, mediumClassic, smallClassic, smallClassicLittleFood
2. GhostType: RandomGhost or DirectionalGhost (Note the DirectionalGhost is a greedyghost that chases Pacman)
4. the --quietTextGraphics flag is passed in to avoid any graphics (The Zoo does not support tkinter so run with quietTextGraphics)
5. If both maxDuration and numgames are passed in, the code will terminate whichever limit is hit first. maxDuration is amount of time in minutes.
5. -a passes in a time limit in seconds for the MCTS Agent for each action. The default is 0.05 seconds
Example to run for 6 mins: 
python3 pacman.py --layout smallClassic -p MCTSAgent -g DirectionalGhost --quietTextGraphics --numGames 20 --maxDuration 5 -a timelimit=0.08
will run the game for either 6 minutes or 20 games and output stats whichever is hit first, where our MCTS agent has 0.08 seconds per move.

Results Analysis:
Games were tested on the mediumClassic Map, smallClassic Map using numGames=20000 games and maxDuration=240 minutes for both maps. Agents were given a 0.05 seconds timelimit. So to replicate results, you should run the following two commands:
    python3 pacman.py --layout smallClassic -p MCTSAgent -g DirectionalGhost --quietTextGraphics --numGames 20000 --maxDuration 240 -a timelimit=0.05
    python3 pacman.py --layout mediumClassic -p MCTSAgent -g DirectionalGhost --quietTextGraphics --numGames 20000 --maxDuration 240 -a timelimit=0.05
The statistics were as follows:
Small Classic:

Medium Classic

Please note the above results were conducted on my personal computer and might have slightly different results when run on the zoo due to differences in computational power as this is a Monte-Carlo process gated by time. Finally there's also elements of noise added to ensure a good exploration exploitation balance.


**Neural Network Agent** (Thomas Chung)\
To train the Neural Network, run:\
`python3 neuralNetworkAgent.py`\
This will train the NN for whatever number of generations specified by the code (right now n=25), but if you switch the else statement of the __main__ function (in neuralNetworkAgent.py) with what is commented out, it will run for a certain amount of time (default = 4 hours).

To run the best Neural Network you have saved (a default is already loaded in in winner.pkl) once, run\
`python3 neuralNetworkAgent.py -best`

To test the Neural Network agent and return the mean & variance over a disclosed amount of time (right now 30 seconds), run\
`python3 neuralNetworkAgent.py -test` --> results will print out in 30 seconds

**Results**\
For the Neural Network, the agent trained for 6 hours and ran through 2242 generations. The factors used were distance to the nearest (only) ghost, the distance 
to the nearest food pellet, and the local pellet density in a 5 tile radius (manhattan distance). We then tested the agent for 4 hours on both Greedy and Random agents.\
Greedy Agent:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Total Games Played: 2876636\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean Score: -223.586781574033\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Variance: 1725.755372139545\
Random Agent:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Total Games Played: 143316\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean Score: -210.09742108347984\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Variance: 10467.106219215242\

**Discussion**
For the Neural Network agent, we had three features, and they were initially:
1. The manhattan distance to the ghost
2. The manhattan distance to the nearest food pellet
3. The overall pellet density
The last feature was seen as incorrect pretty quickly and replaced with the pellet density in a five tile radius. Additionally, to reduce the likelihood of loops,
repeated actions back and forth were banned, the stop movement was banned, and a slight epsilon was added to prevent large loops over time.

However, after these results were measured against the random and greedy ghost, the distance to ghost feature was then replaced with a binary deciding whether the ghost
was within a 3 tile radius to try and prioritize the ghost distance, and the epsilon was removed. Results showed a slight improvement in the game, but ultimately wasn't
enough to just prioritize the ghost the most.

