# MCTS baselines

This is the code to play against three SimpleAgents using Monte-Carlo Tree Search (MCTS). The algorithm is inspired by [AlphaGoZero paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf). The code is using the default forward model in Pommerman, which means it is too slow for use during evaluation. Also as Pommerman is partially observable, it is unclear how to use MCTS in that context. But this does not prevent using it at training time, for example in the manner of [Expert Iteration](https://arxiv.org/abs/1705.08439).

## Running the code

For interactive test:
```
python mcts_agent.py --mcts_iters 10 --num_episodes 1 --num_runners 1 --render
```
For gathering statistics:
```
python mcts_agent.py --mcts_iters 100 --num_episodes 400 --num_runners 16
```
**NB!** `--num_runners` should be the number of cores you have and divisible with 4 (each process evaluates agent at specific position). `--num_episodes` should be divisible by `--num_runners`. I needed at least 400 episodes to get reliable average reward.

## Results

Here are some preliminary results that prove effectiveness of MCTS.

| MCTS iterations | Avg. reward | Avg. length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | ---: | ---: |
| 100 | -0.736 | 187 | 1000 | 6.756 |
| 200 | -0.573 | 230 | 978 | 14.641 |
| 300 | -0.451 | 244 | 430 | 25.037 |
| 400 | -0.362 | 265 | 533 | 32.574 |
| 500 | -0.269 | 279 | 457 | 41.559 |
| 600 | -0.206 | 297 | 456 | 50.209 |
| 700 | -0.188 | 297 | 261 | 60.865 |
| 800 | -0.147 | 297 | 326 | 71.894 |
| 900 | -0.147 | 299 | 265 | 81.209 |
| 1000 | -0.105 | 288 | 400 | 91.769 |

The columns of the table:
* **MCTS iterations** - how many MCTS iterations were used per timestep.
* **Avg. reward** - average reward when running against three SimpleAgents.
* **Avg. length** - average episode length when running against three SimpleAgents.
* **Eval. episodes** - how many episodes were used to calculate previous two.
* **Time per step (s)** - time per timestep in seconds.

Finally some graphs from the same table.

![Average reward vs MCTS iterations](/mcts/images/avg_reward_vs_mcts_iters.png)
![Average episode length vs MCTS iterations](/mcts/images/avg_length_vs_mcts_iters.png)
![Time per step (s) vs MCTS iterations](/mcts/images/time_per_step_vs_mcts_iters.png)
