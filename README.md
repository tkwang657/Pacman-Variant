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
will run the game for either 5 minutes or 20 games and output stats whichever is hit first, where our MCTS agent has 0.08 seconds per move.

Algorithmic Overview:
    Game state is encoded with 2 features: is_ghost_near_me and direction_to_nearest_pellet. This gives 8 possible game states. The statistics are stored along the edges, so effectively there are 40 possible (state, action) pairs.

    a custom value estimator, instead of a random rollout is used for time_efficiency, 
    "self.raw_game_state.getScore()/100-self._distance_to_nearest_pellet(self.data)+10/len(self.data["food_grid"])+ random.randint(0,2)"

Selection of time_limit:
There was insufficient time to run a thorough testing of time_limit given to the MCTS call, but experimentation revealed that more time does not necessarily lead to better performance. After choosing values within the range of (0.01 - 5) seconds, a final time_limit of 0.05 was chosen.

Results Analysis:
We would like to highly stress that the performance of the MCTSAgent is most clearly seen in a visual setting, where we can physically see the moves that the Agent takes to avoid the Ghost. Statistics have are unable to capture how close an agent makes it to the end of the level, so I have gathered some data, namely the foodLeft to show the progression through the level. Through visual analysis over ~100 games, we observed that many instances of the MCTS agent losing was because they ended up in a dead end within the Ghost Spawn.

Games were tested on the mediumClassic Map, smallClassic Map using numGames=10000 games and maxDuration=240 minutes for both maps. Agents were given a 0.05 seconds timelimit. So to replicate results, you should run the following two commands:
    python3 pacman.py --layout smallClassic -p MCTSAgent -g DirectionalGhost --quietTextGraphics --numGames 10000 --maxDuration 240 -a timelimit=0.05
    python3 pacman.py --layout mediumClassic -p MCTSAgent -g DirectionalGhost --quietTextGraphics --numGames 10000 --maxDuration 240 -a timelimit=0.05
Small Classic:
Average Score: -28.722266560255385
Score std:    310.0363321942598
FoodLeft Mean:  12.000798084596967
FoodLeft std: 12.12066887132062
Win Rate:      476/3657 (0.13)

Medium Classic:
Average Score: 839.7
Score std:    499.3724161384968
FoodLeft Mean:  6.05
FoodLeft std: 10.892543321006348
Win Rate:      1516/2167 (0.70)

The agent also seems to do perform significantly better on the MediumClassic map compared to the smallClassic Map. Perhaps this is because the larger state space of MediumClassic map allows for a more generalisability of the game state via our features whereas the small map might be too specific.

Understandably, much less games were played for the mediumClassic compared to the SmallClassic, this makes sense because the mediumClassic map is larger and each playthrough takes more time. There is a high variance in the score compared to the foodLeft metric. This score is calculated by penalizing -500 for collision with ghost, +10 for each food pellet eaten, -1 for second that passes by, +500 for victory. Although it correctly encapsulates the idea that an agent that takes longer to win is worse than one that takes less time to win, it fails to give a notion of performance if the agent loses. Since our agents have a low winning rate, it was more beneficial to gather data on foodLeft, which captures the idea of how far away Pacman is from victory.


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

