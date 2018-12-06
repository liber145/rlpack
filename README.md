### Introduction

**rlpack** is an intuitive, lightweight and flexible reinforcement learning library.
It bundles will solve your problem of where to start with documentation,
by providing a basic explanation of how to do it easily.


**Features:**

- Lightweight
- Decoupling agent and environment, making it easy to integrate and use
- Easy to reproduce up-to-date RL algorithms


### Usage

Look how easy it is to use:


```python
from tqdm import tqdm
import numpy as np
from rlpack.algos import PPO
from rlpack.environment import AsyncAtariWrapper
from rlpack.common import DistributedMemory

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
    $ conda env create -f environment.yml
    $ conda activate py36
```

2. Install `rlpack` by running:

```bash
    $ git clone https://github.com/smsxgz/rl-algo.git
    $ cd rl-algo
    $ python setup.py install
```

It will install a basic learning environment in `gym`.
To install more environments in `gym`, please refer to https://github.com/openai/gym.
