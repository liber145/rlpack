Introduction
============

**rlpack** is an intuitive, lightweight and flexible reinforcement learning library.
It bundles will solve your problem of where to start with documentation,
by providing a basic explanation of how to do it easily.


**Features:**

- Lightweight, depending only on TensorFlow and Numpy
- Decoupling agent and environment, making it easy to integrate and use
- Easy to reproduce up-to-date RL algorithms


Usage
=====

Look how easy it is to use:

.. code:: python

    from tqdm import tqdm
    from rlpack.algos import PPO
    from rlpack.environment import AtariWrapper
    # Get your stuff done

    env = AtariWrapper("BreakoutNoFrameskip-v4")
    pol = PPO()

    obs = env.reset()
    for i in tqdm(range(10000)):
        for _ in range(128):
            actions = agent.get_action(obs)
            next_obs, rewards, dones, infos = env.step(actions)

            memory.store_sard(obs, actions, rewards, dones)
            obs = next_obs

        data_batch = memory.get_last_traj()
        update_ratio = i / 10000
        agent.update(data_batch, update_ratio)



Installation
============

Python3.6+ is required.

1. Install the dependencies using `environment.yml`:

.. code:: bash

    $ conda env create -f environment.yml
    $ conda activate py36


2. Install `rlpack` by running:

.. code:: bash

    $ git clone https://github.com/smsxgz/rl-algo.git
    $ cd rl-algo
    $ python setup.py install


It will install a basic learning environment in `gym`.
To install more environments in `gym`, please refer to https://github.com/openai/gym.

