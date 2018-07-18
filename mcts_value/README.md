# MCTS with value function

[Previously](../mcts_selfplay) I used MCTS without value function. This code introduces some experiments using MCTS guided by both policy and value function. In addition it makes use of [Cython environment](../cython_env) to speed up search.

## Running the code

First download [conv3x256value.h5](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/conv3x256value.h5) and save it in this directory.

For interactive test:
```
python mcts_value_agent.py conv3x256value.h5 --mcts_iters 100 --num_episodes 1 --num_runners 1 --render
```
For gathering statistics:
```
python mcts_value_agent.py conv3x256value.h5 --mcts_iters 100 --num_episodes 400 --num_runners 4
```
**NB!** `--num_runners` should be the number of cores you have and divisible with 4 (each process evaluates agent at specific position). `--num_episodes` should be divisible by `--num_runners`. I needed at least 400 episodes to get reliable average reward.

## Results

### Policy vs value vs discount

The first set of experiments verifies how much contribution is coming from policy and value network, also if bigger discount is really worse in real gameplay.

| Model | Avg. episode reward | Avg. episode length | Avg. rollout length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv3x256 + MCTS100 disc0.9 | 0.445 | 498 | 9.017 | 400 | 0.462 |
| Conv3x256 + MCTS100 disc0.9 policy only | 0.325 | 403 | 9.481 | 400 | 0.462 |
| Conv3x256 + MCTS100 disc0.9 value only | -0.350 | 393 | 1.952 | 400 | 0.459 |
| Conv3x256 + MCTS100 disc0.99 | 0.275 | 509 | 11.901 | 400 | 0.462 |

Observations:
 * Using both policy and value network boosts performance considerably.
 * Using prior policy is more important than using value function.
 * Bigger discount (credit assignment over longer horizons) does not help.

### Model speed

The main bottleneck is model speed, so I tried couple of bigger and smaller models to see how it affects the result.

| Model | Avg. episode reward | Avg. episode length | Avg. rollout length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Conv3x256 + MCTS100 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/conv3x256value.h5) | 0.445 | 498 | 9.017 | 400 | 0.462 |
| [AGZ20x64 + MCTS100 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/agz20x64value.h5) | 0.520 | 431 | 13.137 | 400 | 2.194 |
| [Conv3x32 + MCTS100 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/conv3x32value.h5) | 0.449 | 445 | 10.014 | 385 | 0.374 |
| [Linear + MCTS100 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/linearvalue.h5) | -0.460 | 385 | 2.686 | 400 | 0.274 |

Observations:
 * Better model sometimes translates into better performance (AGZ).
 * Sometimes smaller and worse model can achieve comparable performance to bigger model (Conv3x32), i.e. MCTS can compensate worse model.
 * On the other hand for very bad model (Linear) MCTS compensation is not enough.
 * It is not clear which model would win if the time limit of 100ms is considered. Better but slower AGZ model might end up worse than faster but otherwise inferior Conv3x32 model.

### Alternatives to learned value function

Instead of using learned value function one could use Monte-Carlo rollouts (play using random actions till the end of game), rollouts using some simple policy or hand-crafted evaluation function. Here are some experiments with those ideas.

| Scheme | Avg. episode reward | Avg. episode length | Avg. rollout length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv3x256 + MCTS100 reward as value function | 0.325 | 403 | 9.481 | 400 | 0.462 |
| Conv3x256 + MCTS100 MC rollout | -0.070 | 476 | 3.729 | 400 | 0.564 |
| Conv3x256 + MCTS100 MC rollout 8 steps | 0.090 | 499 | 7.784 | 400 | 0.549 |
| Conv3x256 + MCTS100 policy rollout 8 steps | 0.460 | 468 | 11.936 | 400 | 4.761 |

Schemes:
* **reward as value function** - use current rewards at leaf node as value function. This returns mostly zeros, unless there is obvious chance of getting killed or winning during running tree policy. Works surprisingly well.
* **MC rollout** - uses random moves to play till the end of game. This tends to result in very noisy value estimate, because agents just randomly set up bombs and get killed. Use of this noisy value estimate kills the performance.
* **MC rollout 8 steps** - motivated by problems with previous scheme, here we just do random rollout for 8 steps (it takes 9 steps for bomb to explode) and then use current reward as value function. Works barely better.
* **policy rollout 8 steps** - use current network policy to do rollout for 8 steps and then use rewards as value function. Doing full rollout till the end of game turned out to be too slow. Achieves surprisingly good performance, but takes major hit in time per step.

## Final remarks

While average reward 0.5 against three SimpleAgents is encouraging (this means average win rate of 75%, above 68% of [agent that won the first competition](https://yichengong.github.io/)), and probably could be increased with more iterations, there are couple of caveats one should be aware of:
 * In this implementation MCTS has full access to game state, for example it knows what item is behind each wooden wall. While I haven't noticed agent making use of that information, this is definitely an advantage. Handling the partial observability in Pommerman with MCTS is open question.
 * The code uses still 400ms to make actions. While 100ms seems achievable by batching forward passes of network, it is not clear how it affects MCTS performance - parallel iterations cannot make use of node updates from other iterations.
 * The agent may very well be overfitting to play against three SimpleAgents. While it uses self-play, the self-play agents use SimpleAgent imitation model to guide their moves, therefore their actions resemble pretty much what SimpleAgents might do. It's not clear if this would generalize to other agents.
