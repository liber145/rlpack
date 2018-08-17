import os
import zmq
import time
import msgpack
import numpy as np
import msgpack_numpy
from gym import spaces
from collections import OrderedDict
from middleware.log import logger


msgpack_numpy.patch()
INITED_RAY = False


def _get_multi_space(space, n):
    if type(space) == spaces.Box:
        return spaces.Box(
            low=np.repeat([space.low], n, axis=0),
            high=np.repeat([space.high], n, axis=0),
            dtype=space.dtype)
    elif type(space) == spaces.Discrete:
        return spaces.MultiDiscrete(nvec=[space.n] * n)
    else:
        # TODO: add other spaces class
        raise NotImplementedError


class EnvMaker:
    def __init__(self, num_agents, env_fn, basename, backend='ray'):
        self._num_agents = num_agents

        env = env_fn()
        self._redefine_space(env)
        env.close()

        base_path = './.ipc'
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        url_path = os.path.join(base_path, '{}-Agent.ipc'.format(basename))
        if os.path.exists(url_path):
            os.remove(url_path)

        url = 'ipc://{}'.format(url_path)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.bind(url)

        self._run_subagents(env_fn, url, backend)

        self.addrs = OrderedDict()

        # Waiting for message b'ready' from each subagent
        for _ in range(self._num_agents):
            addr, empty, msg = self._socket.recv_multipart()
            self.addrs[addr] = None
            assert msg == b'ready'
        print('All subagents are ready! ')

    def _run_subagents(self, env_fn, url, backend='ray'):
        if backend == 'ray':
            import ray
            from .sub_env import sub_env

            ray_subagent = ray.remote(sub_env)

            global INITED_RAY
            if not INITED_RAY:
                ray.init()
                INITED_RAY = True

            [
                ray_subagent.remote(env_fn, i, url)
                for i in range(self._num_agents)
            ]

        elif backend == 'multiprocessing':
            from multiprocessing import Process
            from .sub_env import sub_env
            [
                Process(target=sub_env, args=(env_fn, i, url)).start()
                for i in range(self._num_agents)
            ]

        else:
            raise NotImplementedError

    def _redefine_space(self, sample_env):
        self.observation_space = _get_multi_space(sample_env.observation_space,
                                                  self._num_agents)

        self.action_space = _get_multi_space(sample_env.action_space,
                                             self._num_agents)

    def reset(self):
        for addr in self.addrs:
            self._socket.send_multipart([addr, b'', b'reset'])
        for _ in range(self._num_agents):
            addr, empty, msg = self._socket.recv_multipart()
            msg = msgpack.loads(msg)
            self.addrs[addr] = msg
        return np.array(list(self.addrs.values()))

    def step(self, actions):
        for action, addr in zip(actions, self.addrs.keys()):
            self._socket.send_multipart([addr, b'', msgpack.dumps(action)])

        info = {}
        for _ in range(self._num_agents):
            addr, empty, msg = self._socket.recv_multipart()
            msg = msgpack.loads(msg)
            if msg[-1]:
                info[addr] = msg[-1]
            self.addrs[addr] = msg[:-1]
        states, rewards, dones = map(np.array, zip(*self.addrs.values()))
        return states, rewards, dones, info

    def close(self):
        for addr in self.addrs:
            self._socket.send_multipart([addr, b'', b'close'])
        time.sleep(1)
        self._socket.close()
        self._context.term()
