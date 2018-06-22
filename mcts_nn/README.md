# MCTS + NN baselines

This is the code to play against three SimpleAgents using Monte-Carlo Tree Search (MCTS) with neural network guidance. The code is basically the same as in [pure MCTS](../mcts), just uses [imitation network](../imitation) to initialize action probabilities.

## Running the code

For interactive test:
```
python mcts_agent_nn.py conv256.h5 --mcts_iters 10 --num_episodes 1 --num_runners 1 --render
```
For gathering statistics:
```
python mcts_agent_nn.py conv256.h5 --mcts_iters 100 --num_episodes 400 --num_runners 16
```
**NB!** `--num_runners` should be the number of cores you have and divisible with 4 (each process evaluates agent at specific position). `--num_episodes` should be divisible by `--num_runners`. I needed at least 400 episodes to get reliable average reward.

## Results

Clearly making use on neural network guidance improves MCTS - we have the first positive result against three SimpleAgents!

| Model | Avg. reward | Avg. length | Eval. episodes | Time per step (s) |
| --- | ---: | ---: | ---: | ---: |
| [Conv256](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/conv256.h5) | -0.63 | 299 | 400 |  |
| MCTS100 | -0.736 | 187 | 1000 | 6.756 |
| MCTS200 | -0.573 | 230 | 978 | 14.641 |
| Conv256 + MCTS50 | -0.122 | 554 | 337 | 5.163 |
| Conv256 + MCTS100 | -0.013 | 555 | 158 | 12.040 |
| Conv256 + MCTS200 | 0.053 | 576 | 76 | 26.076 |

The columns of the table:
* **Model** - model used, either only network, only MCTS or combined.
* **Avg. reward** - average reward when running against three SimpleAgents.
* **Avg. length** - average episode length when running against three SimpleAgents.
* **Eval. episodes** - how many episodes were used to calculate previous two.
* **Time per step (s)** - time per timestep in seconds.

## Performance

The idea of [Expert Iteration](https://arxiv.org/abs/1705.08439) is to train a better neural network by using MCTS action probabilities as targets. For this we would need to gather another dataset using MCTS + NN. As can be seen from step times above, that would be rather slow. Also we would probably benefit from more MCTS iterations. This motivates investigation into improving the performance of the code. For this I ran profiling on runner process.

How to run profiling:
```
python mcts_agent_nn.py conv256.h5 --num_runners 1 --num_episodes 1 --mcts_iters 10 --profile conv256_mcts10.prof
```

Then I used excellent [Snakeviz tool](https://jiffyclub.github.io/snakeviz/) to visualize the results (click on the image for interactive version):

[![Profiling results](/mcts_nn/profiling/conv256_mcts10.png)](/mcts_nn/profiling/conv256_mcts10.html)

Analysis:
* In current version of the code most of the time is spent on running SimpleAgents, in particular in [\_djikstra()](https://github.com/MultiAgentLearning/playground/blob/master/pommerman/agents/simple_agent.py#L110-L169) method. The solution would be to use self-play instead, which was actually the plan all along. The problem with self-play though is that reasonably good agents that are able to avoid bombs might get stuck in local maxima of doing nothing - why lay bombs and take the risk of blowing yourself up if just sitting in the corner and doing nothing achieves the same -1 reward. This needs to be verified though.
* Next to that is `predict()` method of network. What can be done here is batching the predictions from different MCTS iterations. This might complicate the code considerably.
* Then environment `step()` method is the next culprit. The idea is to rewrite forward model in [C++](https://github.com/MultiAgentLearning/playground/issues/103) or [Cython](http://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html). Cython approach might be easier to implement and integrate with existing Python code, but possibly not as performant as complete rewrite in C++. Also notice that `get_observations()` method of forward model takes considerable amount of time. When doing self-play, we might not need observations at every step at all, so this could be skipped.
* `get_json_info()` and `reset()` environment methods are the next. Both come down to non-optimal state representation. To be honest using JSON state in MCTS was more of a hack, but it served well for the proof-of-concept. Converting Pommerman internal mix of objects and tables into either JSON state or Numpy observation incurs considerable overhead. When rewriting the forward model, I would keep the internal state either as Numpy matrices (so it would be easy to convert it to observation that has to fed to the network), or low-footprint byte array that can be used as index in tree-dictionary in its raw form.
* Finally `action()` method of `MCTSNode` is used a lot during tree search and using `argmax_tiebreaking()` instead of normal `np.argmax()` seems to have some penalty. It might be possible to bring back `np.argmax()` in `action()` method (but not in `probs()`!), but how it affects the learning performance remains to be seen.
