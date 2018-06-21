# Imitation learning baselines

This is the code to train agents to imitate SimpleAgent from Pommerman repository. The scripts can be divided into three groups: data collection, model training and model evaluation.

## Data collection

* `collect_simple.py` - run four SimpleAgents and save their observations, actions and rewards.
* `Discount.ipynb` - copy final episode reward to all timesteps, optionally discounted.
* `Clean.ipynb` - remove repeating actions and observations when SimpleAgent gets stuck. Improves class balance of the dataset. NB! Must be run **after** Discount.ipynb!

How to run data collection:
```
python collect_simple.py --num_episodes 10 --render test_10.npz
```
I collected dataset of 600 episodes for training and 100 episodes for validation. The dataset can be downloaded from [here](https://github.com/tambetm/pommerman-baselines/releases/tag/simple_600K).

## Model training

* `Linear model.ipynb` - linear model. Performs extremely poorly.
* `Fully connected model.ipynb` - fully connected model with two hidden layers with 128 units + additional hidden layer with 128 units for value head. Performs very poorly.
* `Conv model.ipynb` - simple convolutional network with three 3x3 convolutions with 32 filters + fully connected layer with 128 nodes + another fully connected layer with 128 units for value head. Performs poorly.
* `AlphaGoZero model.ipynb` - model inspired by [AlphaGoZero paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf). Performs well, but tends to overfit.
* `Conv256 model.ipynb` - three 3x3 convolutions with 256 filters, no fully connected layers. Achieves better validation set performance as AlphaGoZero, but overfits less.

Keras + Tensorflow was used to train the models.

## Model evaluation
* `eval_model.py` - evaluates model against three SimpleAgents and reports average reward, average length and time per timestep.

How to run evaluation:
```
python eval_model.py --num_episodes 4 --render conv256.h5
```
**NB!** `--num_episodes` should be divisible by 4, because the agent is evaluated at each of the four positions.

## Results

| Model | Avg. reward | Avg. length |  Val. acc. | Val. exp. var. | Train acc. | Train exp. var. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleAgent | -0.61 | 258 | | | | |
| StopAgent | -0.85 | 291 | | | |
| [Linear](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/linear.h5) | -0.905 | 110 | 0.363 | -0.006 | 0.393 | 0.013 |
| [Fully connected](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/dense.h5) | -0.94 | 157 | 0.392 | | 0.400 | |
| [Conv](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/conv.h5) | -0.755 | 262 | 0.630 | -0.391 | 0.661 | 0.883 |
| [AlphaGoZero](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/AGZ.h5) | -0.635 | 278 | 0.668 | -0.278 | 0.761 | 0.930 |
| [Conv256](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/conv256.h5) | -0.63 | 299 | 0.676 | | 0.705 | |

The columns of the table:
* **Model** - click on model name to download weights.
* **Avg. reward** - average reward when running against three SimpleAgents.
* **Avg. length** - average episode length when running against three SimpleAgents.
* **Val. acc.** - validation set accuracy of action prediction.
* **Val. exp. var.** - validation set explained variance of value prediction.
* **Train. acc.** - training set accuracy of action prediction.
* **Train. exp. var.** - training set explained variance of value prediction.

Two first models are baselines - SimpleAgent playing against itself and StopAgent which always takes Stop action. The best models are pretty close to SimpleAgent performance, which they tried to imitate. Action prediction accuracy 100% is impossible to achieve because SimpleAgent chooses some of the actions randomly.

Some of the models were trained without value head by setting its loss weight to zero, for those the explained variance is not reported. Not too much effort was spent on tuning hyperparameters, you are welcome to submit better models (with appropriate statistics).

Clearly the models tend to overfit on value prediction. This can be explained by low number of episodes in the training set - there are only 600 episodes. For example this means that there are only 600 different configurations of stone walls. I'm collecting a dataset with one sample per episode which hopefully improves this.
