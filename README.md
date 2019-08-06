### 本包简介
----------------------

**rlpack**是一个基于**tensorflow**的强化学习算法库，解耦算法和环境，方便调用。

<!--
is an intuitive, lightweight and flexible reinforcement learning library based on TensorFlow.
It bundles up-to-date reinforcement learning algorithms. 
-->


### 使用方法
----------------------
下面展示如何使用`rlpack`在[`MuJoCo`](https://github.com/openai/mujoco-py)环境中运行`PPO`算法。

```python
# -*- coding: utf-8 -*-


import argparse
import time
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

from rlpack.algos import PPO
from rlpack.utils import mlp, mlp_gaussian_policy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  type=str, default="Reacher-v2")
args = parser.parse_args()

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'early_stop', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))


def policy_fn(x, a):
    return mlp_gaussian_policy(x, a, hidden_sizes=[64, 64], activation=tf.tanh)


def value_fn(x):
    v = mlp(x, [64, 64, 1])
    return tf.squeeze(v, axis=1)


def run_main():
    env = gym.make(args.env)
    dim_obs = env.observation_space.shape[0]
    dim_act = env.action_space.shape[0]
    max_ep_len = 1000

    agent = PPO(dim_act=dim_act, dim_obs=dim_obs, policy_fn=policy_fn, value_fn=value_fn, save_path="./log/ppo")

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(50):
        memory, ep_ret_list, ep_len_list = Memory(), [], []
        for t in range(1000):
            a = agent.get_action(o[np.newaxis, :])[0]
            nexto, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            memory.push(o, a, r, int(d), int(ep_len == max_ep_len or t == 1000-1), nexto)

            o = nexto

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == 1000-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                if terminal:
                    # 当到达完结状态或是最长状态时，记录结果
                    ep_ret_list.append(ep_ret)
                    ep_len_list.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        print(f"{epoch}th epoch. average_return={np.mean(ep_ret_list)}, average_len={np.mean(ep_len_list)}")

        # 更新策略。
        batch = memory.sample()
        agent.update([np.array(x) for x in batch])

    elapsed_time = time.time() - start_time
    print("elapsed time:", elapsed_time)


if __name__ == "__main__":
    run_main()
```


### 安装流程
----------------------

1. 安装依赖包

安装所需依赖软件包，请看`environment.yml`.
建议使用[`Anaconda`](https://www.anaconda.com/distribution/)配置`python`运行环境，可用以下脚本安装。

```bash
    $ git clone https://github.com/liber145/rlpack
    $ cd rlpack
    $ conda env create -f environment.yml
    $ conda activate py36
```

2. 安装`rlpack`

```bash
    $ python setup.py install
```

以上流程会安装一个常用的强化学习运行环境[`gym`](https://github.com/openai/gym).
该环境还支持一些复杂的强化学习环境，比如[`MuJoCo`](https://github.com/openai/mujoco-py)，具体请看[`gym`](https://github.com/openai/gym)的介绍。

### 算法列表
----------------------


| 算法  | 论文链接                                                                                             | 类型        | 连续动作  | 离散动作  |
|------|-----------------------------------------------------------------------------------------------------|------------|----------|----------|
| TRPO | [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)                                | on-policy  | &#10004; | &#10005; |
| PPO  | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                         | on-policy  | &#10004; | &#10005; |
| TD3  | [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) | off-policy | &#10004; | &#10005; |
| DDPG | [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)             | off-policy | &#10004; | &#10005; |
| SAC  | [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)                                               | off-policy | &#10004; | &#10005; |



- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)



### 参考代码
----------------------
在实现过程中，参考了其他优秀代码，帮助比较大的列举如下：
- [openai的baselines](https://github.com/openai/baselines)
- [openai的spinningup](https://github.com/openai/spinningup)
- [清华张楚珩同学的算法实现](https://github.com/zhangchuheng123/Reinforcement-Implementation)



### 学习资料
----------------------

- [Introduction to Reinforcement Learning](https://dl.acm.org/citation.cfm?id=551283)