# Learning value function using imitation

In my [previous experiments](../imitation) with imitation learning I failed to learn value function. Turns out [the previous dataset](https://github.com/tambetm/pommerman-baselines/releases/tag/simple_600K) was not diverse enough - it consisted about 600K observation-value pairs, but these were collected only from 600 episodes. For example this means that the model only saw 600 different configurations of stone walls. That clearly was not enough to fit value model, but apparently was enough to fit policy model.

To overcome those issues I collected [a new dataset](https://github.com/tambetm/pommerman-baselines/releases/tag/single_600K) that includes only one observation from each agent from each episode (inspired by [Expert Iteration paper](https://arxiv.org/abs/1705.08439)). This produces enough diversity to learn value function, but explained variance still remained quite low. Turns out you can get much better explained variance by using smaller discount rate (i.e. 0.9). Arguably Pommerman does not need very long-term strategy or SimpleAgent policy just does not have long-term strategy.

Together with [Single sample per episode 600K dataset](https://github.com/tambetm/pommerman-baselines/releases/tag/single_600K) I also release [bunch of models](https://github.com/tambetm/pommerman-baselines/releases/tag/single_600K_models) trained using this dataset. These models achieve the same action prediction accuracy as [previous models](https://github.com/tambetm/pommerman-baselines/releases/tag/simple_600K_models), but also include value prediction, which improves MCTS results considerably, as can be seen from accompanied [MCTS experiments](../mcts_value).

## Dataset collection

In case you want to re-run the dataset collection:
```
python collect_single.py --num_episodes 40000 --discount 0.9 single_600K_disc0.9.npz
```

In practice it may be worth running number of processes in parallel and later concatenate their results.

## Results

### Datasets

The first set of experiments showcases the need for better dataset and lowered discount rate.

| Model | Validation explained variance | Train explained variance |
| --- | ---: | ---: |
| Conv3x32 original dataset disc0.99 | -0.190 | 0.323 |
| Conv3x32 one-sample-per-episode dataset disc0.99 | 0.256 | 0.280 |
| Conv3x32 one-sample-per-episode dataset disc0.9 | 0.594 | 0.633 |

Observations:
 * Training model on original dataset overfits a lot, the result is practically unusable as shown by validation set explained variance.
 * Using one sample per episode produces enough diversity to make model generalize.
 * Using smaller discount rate achieves even better explained variance. Whether this is property of game or policy is unclear. Or as discount 0.9 produces values close to 0 for most observations, maybe just smaller values make it easier? 

### Multitask training

The second set of experiments verifies if the model performance remains the same when trained to predict both actions and values.

| Model | Validation accuracy | Validation explained variance | Train accuracy | Train explained variance |
| --- | ---: | ---: | ---: | ---: |
| Conv3x32 value only | | 0.594 | | 0.633 |
| Conv3x32 policy only | 0.755 | | 0.783 | |
| Conv3x32 policy and value | 0.753 | 0.508 | 0.776 | 0.530 |
| Conv3x32 policy and value coef10 | 0.755 | 0.581 | 0.776 | 0.625 |

Observations:
 * Policy and value network can be successfully trained together and achieve performance comparable to training only value network.
 * Value loss coefficient needs to be increased to match the single task performance.

### Bigger models

The third set of experiments verifies if better results can be achieved with bigger models. In addition to imitation training results the average reward and average episode length against three SimpleAgents is shown. 

| Model | Avg. reward | Avg. length | Time per step | Val. acc. | Val. exp. var. | Train acc. | Train exp. var. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ***Baselines*** |
| SimpleAgent | -0.61 | 258 | | | | | |
| [Conv3x256 policy only using Simple 600K dataset](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/conv256.h5) | -0.630 | 299 | | 0.676 | | 0.705 | |
| ***Imitation with value*** |
| [Conv3x256 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/conv3x256value.h5) | -0.630 | 274 | 0.009 | 0.758 | 0.601 | 0.776 | 0.662 |
| [Conv3x256 disc0.99](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/conv3x256value_disc0.99.h5) | -0.760 | 242 | 0.009 | 0.752 | 0.264 | 0.764 | 0.291 |
| [AGZ20x32 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/agz20x32value.h5) | -0.705 | 251 | 0.011 | 0.755 | 0.613 | 0.773 | 0.659 |
| [AGZ20x64 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/agz20x64value.h5) | -0.715 | 243 | 0.014 | 0.764 | 0.642 | 0.787 | 0.728 |
| [Conv3x32 disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/conv3x32value.h5) | -0.785 | 243 | 0.008 | 0.755 | 0.581 | 0.776 | 0.625 |
| [Linear disc0.9](https://github.com/tambetm/pommerman-baselines/releases/download/single_600K_models/linearvalue.h5) | -0.980 | 92 | 0.007 | 0.452 | 0.307 | 0.455 | 0.310 |

Observations:
 * Conv3x256 performance against three SimpleAgents matches the previous model trained on Simple 600K dataset.
 * Bigger model does not help much with 0.99 discount.
 * Using AlphaGoZero inspired model results in minor improvements in performance and it is unclear, if those weight over the increased time per step.
 * Better action prediction accuracy does not mean better performance against three SimpleAgents.
 * Validation set accuracy for Simple 600K dataset is lower, this might be caused by using different, cleaned validation set. Cleaned validation set does not include repeated actions, which might make prediction harder.

In the end I included couple of very simple and fast models. Links can be used to download corresponding weights.

Continues with [MCTS experiments](../mcts_value).
