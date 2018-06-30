# MCTS selfplay

In Pommerman four players decide their moves simultaneously, not sequentially as in Go or chess. This makes it challenging to adapt standard MCTS self-play techniques to Pommerman. This code does MCTS self-play by treating moves of four players as on big compound action. Notice that this increases the branching rate considerably - instead of 6 outgoing edges from each state, we now have 6x6x6x6 = 1296 outgoing edges.

## Running the code

First download [conv256.h5](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/conv256.h5) and save it in this directory.

For interactive test:
```
python mcts_selfplay_agent.py conv256.h5 --mcts_iters 10 --num_episodes 1 --num_runners 1 --render
```
For gathering statistics:
```
python mcts_selfplay_agent.py conv256.h5 --mcts_iters 100 --num_episodes 400 --num_runners 16
```
**NB!** `--num_runners` should be the number of cores you have and divisible with 4 (each process evaluates agent at specific position). `--num_episodes` should be divisible by `--num_runners`. I needed at least 400 episodes to get reliable average reward.

## Results without NN

These are the results of vanilla MCTS + selfplay, without network guidance (compare to [normal MCTS](../mcts#results)).

| MCTS iterations | Avg. episode reward | Avg. episode length | Mean rollout length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | --: | ---: | ---: |
| 100 | -0.570 | 213 | 2.06 | 400 | 0.193 |
| 200 | -0.530 | 244 | 2.46 | 400 | 0.437 |
| 300 | -0.540 | 217 | 2.66 | 400 | 0.700 |
| 400 | -0.580 | 239 | 2.83 | 400 | 0.972 |
| 500 | -0.460 | 236 | 2.99 | 400 | 1.263 |
| 600 | -0.535 | 235 | 3.08 | 400 | 1.562 |
| 700 | -0.450 | 271 | 3.25 | 400 | 1.891 |
| 800 | -0.505 | 267 | 3.20 | 400 | 2.164 |
| 900 | -0.410 | 285 | 3.36 | 400 | 2.497 |
| 1000 | -0.422 | 266 | 3.43 | 367 | 2.829 |

The columns of the table:
* **MCTS iterations** - how many MCTS iterations were used per timestep.
* **Avg. episode reward** - average reward when running against three SimpleAgents.
* **Avg. episode length** - average episode length when running against three SimpleAgents.
* **Mean rollout length** - average number of steps till leaf node in MCTS.
* **Eval. episodes** - how many episodes were used to calculate previous two.
* **Time per step (s)** - time per timestep in seconds.

To make the comparison easier, here are some figures:

![Average reward vs MCTS iterations](/mcts_selfplay/images/avg_reward_vs_mcts_iters.png)

As can be seen, with MCTS self-play the average reward does not increase with iterations the same way as when doing MCTS against SimpleAgents. This can be explained by huge branching factor - at each timestep there are 1296 possible outcomes, after two steps that makes 1679616 potential states. As MCTS agents choose their actions initially randomly, the chance is very low that the state they end up after two moves is already in the tree. This is confirmed by average rollout length which was 2-3.5 for self-play and 6-8.5 against SimpleAgents. Such short rollouts do not produce enough knowledge for agents to make informed decisions.

![Average episode length vs MCTS iterations](/mcts_selfplay/images/avg_length_vs_mcts_iters.png)

Because of limited exploration the agents often make random moves, which shows in shorter episodes. They just set up bomb in wrong place or walk into flames. 

![Time per step (s) vs MCTS iterations](/mcts_selfplay/images/time_per_step_vs_mcts_iters.png)

Self-play improves performance a lot though. The key is getting rid of really slow SimpleAgent implementation inside the MCTS iterations.

## Results with NN

These are the results of MCTS + selfplay using network guidance (compare to [MCTS + NN](../mcts_nn#results)).

| Model | Avg. episode reward | Avg. episode length | Mean rollout length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ***Baselines*** |
| [Conv256 + MCTS100](../mcts_nn#results) | -0.102 | 573 | 9.99 | 421 | 11.771 |
| [Conv256 + MCTS200](../mcts_nn#results) | 0.000 | 559 | 11.27 | 340 | 27.753 |
| ***MCTS self-play*** |
| Conv256 + MCTS100 self-play (CPU) | 0.015 | 472 | 11.03 | 400 | 12.219 |
| Conv256 + MCTS500 self-play (GPU) | 0.215 | 482 | 17.00 | 502 | 7.259 |
| Conv256 + MCTS1000 self-play (GPU) | 0.358 | 439 | 17.58 | 187 | 15.675 |

Neural network seems to narrow down the branching rate considerably. Faster rollouts allow for more iterations, which in turn produce strong positive average rewards. The average rollout length was 10-20. Still I find it slightly surprising that MCTS self-play with 100 iterations resulted in better score than MCTS against SimpleAgents with the same number of iterations. But at the same time the network was trained to imitate SimpleAgents, so MCTS rollouts were really played against SimpleAgent-like opponent. Maybe self-play produces slightly more robust moves. In the end this is promising, but of course step times are nowhere near being usable during evaluation.
