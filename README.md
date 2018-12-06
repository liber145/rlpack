### Introduction

**rlpack** is an intuitive, lightweight and flexible reinforcement learning library.
It bundles will solve your problem of where to start with documentation,
by providing a basic explanation of how to do it easily.


**Features:**

- Lightweight
- Decoupling agent and environment, making it easy to integrate and use
- Easy to reproduce up-to-date RL algorithms


### Usage

Look how to use it:


```python
from tqdm import tqdm
import numpy as np
from rlpack.algos import PPO
from rlpack.environment import AsyncAtariWrapper
from rlpack.common import DistributedMemory

# initialization.
env = AsyncAtariWrapper("BreakoutNoFrameskip-v4")
class Config:
    def __init__(self):
        self.n_env = 4
        self.entropy_coef = 0.01
        self.vf_coef = 0.1
        self.trajectory_length = 128
        self.clip_schedule = lambda x: (1 - x) * 0.1
        self.dim_observation = env.dim_observation
        self.dim_action = env.dim_action
config = Config()
agent = PPO(config)
memory = DistributedMemory(10000)
memory.register(env)
epinfos = []

# training process.
obs = env.reset()
memory.store_s(obs)
for i in tqdm(range(10000)):
    for _ in range(config.trajectory_length):
        actions = agent.get_action(obs)
        memory.store_a(actions)
        obs, rewards, dones, infos = env.step(actions)
        memory.store_rds(rewards, dones, obs)

        epinfos.extend([info["episode"] for info in infos if "episode" in info])

    update_ratio = i / 10000
    data_batch = memory.get_last_n_samples(config.trajectory_length)
    agent.update(data_batch, update_ratio)
    print("eprewmean:", np.mean([info["r"] for info in epinfos]))
```


### Installation

Python3.6+ is required.

1. Install the dependencies using `environment.yml`:

```bash
    $ git clone https://github.com/smsxgz/rl-algo.git
    $ cd rl-algo
    $ conda env create -f environment.yml
    $ conda activate py36
```

2. Install `rlpack` by running:

```bash
    $ python setup.py install
```

It will install a basic learning environment in `gym`.
To install more environments like mujoco, please refer to https://github.com/openai/gym.

### Reference

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Introduction to Reinforcement Learning](https://dl.acm.org/citation.cfm?id=551283)
