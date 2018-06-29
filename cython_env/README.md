# Cython environment

This is Cython implementation of [Pommerman environment](https://github.com/MultiAgentLearning/playground). It is based on the original implementation, I only converted it to Cython, annotated variables with C types and rewrote some parts to get rid of dictionaries and other expensive Python types. It achieves **19-20x** speedup compared to the original environment. The key was getting rid of inefficient observation structure and producing directly features that can be fed to neural network.

## Installation

In this directory run:
```
python setup.py develop
```
Probably you will want development install so that you can experiment with changes. To recompile the source after making the changes:
```
python setup.py build_ext --inplace
```
When compiling Cython produces annotated source files in HTML format which can be really useful in determining the parts of code that need optimization. These can be found in `cpommerman` directory after compilation. I made an effort to optimize only the main loop, i.e. the `step()` method and friends. Therefore map generation is mostly untouched, apart from changes needed to adapt to new data types.

## Using

Simplest script to run the environment:
```
import cpommerman
import numpy as np

env = cpommerman.make()

for i in range(1000):
    env.reset()
    done = False
    while not done:
        #state = env.get_state()
        #obs = env.get_observations()
        features = env.get_features()

        # use features, observations or state to produce action
        actions = np.random.randint(6, size=4, dtype=np.uint8)

        # step environment and collect rewards and terminal flag
        env.step(actions)
        rewards = env.get_rewards()
        done = env.get_done()
```
As you can see there are couple of changes to the original environment:
* There is no environment name argument to `make()` function, only `PommeFFACompetition-v0` is currently supported. Also you do not supply the list of agents, as taking actions happens outside of the environment.
* `reset()` and `step()` do not return observation, it's up to you whether you actually need it. There are accessor methods `get_observations()`, `get_rewards()` and `get_done()` to fetch the same information.
* There are couple of additional methods: `get_features()` returns observation as Numpy array to be fed to neural network, `get_state()` return game state in compact binary form (similar to `get_json_info()` in original environment).
* Actions must be Numpy array with `uint8` datatype. Again, this is done for efficiency reasons. The rewards are also returned as Numpy array, not list.
* There is no `render()` method. See `example_nn.py` for a trick how to render a game state in original environment.

Otherwise it should behave exactly the same.

There is example in this directory that showcases the use of `get_features()` together with neural network. First download [conv256.h5](https://github.com/tambetm/pommerman-baselines/releases/download/simple_600K_models/conv256.h5) and save it in this directory. Then run:

```
python example_nn.py conv256.h5 --num_episodes 1 --render
```
You should see neural network playing against itself. As features are nicely stacked, the actions for all four agents can be predicted with one forward pass. Yes, they tend to get stuck quite often, but they were trained to imitate SimpleAgents, so eventually the state distribution diverges from what they have seen.

## Testing

There is a test script that runs original environment and Cython environment in parallel and verifies that the results match. To run the test script:

```
python test.py
```

I have run this script for hundreds of episodes without error, so hopefully they indeed match. Notice that it is not enough to use random moves to test two environments, because random agents tend to die quickly and do not produce more advanced environment behavior.

## Performance

I made some informal performance tests with included `example.py` script. Basically I commented in and out different method invocations and recorded the number of steps per second. I only ran each test once, so these are not statistically valid results, but give some idea what to expect.


| Methods | Steps per second | Reduction | Percentage |
| --- | ---: | ---: | ---: |
| only step() and get_done()       | 41175 |  |  |
| + get_state()                    | 39493 | 1681 | 0.04 |
| + get_observations()             | 23942 | 17233 | 0.42 |
| + get_features()                 | 30685 | 10489 | 0.25 |
| + get_rewards()                  | 37985 | 3190 | 0.08 |
| + get_features() + get_rewards() | 29999 | 11175 | 0.27 |
| + get_state() + get_rewards()    | 36350 | 4825 | 0.12 |
| example_nn.py + conv256.h5       | 286 | 40889 | 0.99 |

As you can see, neural network becomes the main bottleneck.
