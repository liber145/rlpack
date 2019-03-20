"""
A wrapper for distributed asynchronous RL.

The wrapper contains two head, the `AgentWrapper` and `EnvironmentWrapper`.

"""

import logging
import time
from collections import defaultdict, deque
from multiprocessing.managers import BaseManager
from queue import Empty, Queue
from threading import Lock, Thread
from uuid import uuid4

import numpy as np


class AgentWrapper(Thread):
    """Provide agents with data from the environment."""

    def __init__(self, port):
        """Prepare the queue for shared memory.

        Args:
            port: int, enable all environments to connect to.
            reward_keep_episodes:   int, statistics for recent reward.
        """
        super().__init__()

        self._srd_queue = defaultdict(Queue)
        self._action_queue = defaultdict(Queue)
        self._sard_running = defaultdict(list)
        self._sard_finish = []
        self._port = port
        self._agent_server = None
        self._lock = Lock()

        self._prepare_shared_memory_queue()

    def _prepare_shared_memory_queue(self):

        class InferenceMemoryManager(BaseManager):
            pass

        InferenceMemoryManager.register(
            'get_srd', callable=lambda x: self._srd_queue[x])
        InferenceMemoryManager.register(
            'get_a', callable=lambda x: self._action_queue[x])

        m = InferenceMemoryManager(address=('', self._port), authkey=b'secret')
        self._agent_server = m.get_server()

    def _get_n_srd(self, n):
        srds = []
        env_ids = []
        m = 0
        while m < n:
            all_envs = list(self._srd_queue.keys())
            for t in all_envs:
                try:
                    s, r, d = self._srd_queue[t].get(block=False)
                    env_ids.append(t)
                    srds.append((s, r, d))

                    if t in self._sard_running:
                        self._sard_running[t][-1][2] = r
                        self._sard_running[t][-1][3] = d
                        self._sard_running[t].append([s, None, None, None])
                        if d:
                            episode = self._sard_running.pop(t)
                            self._sard_finish.append(episode)
                    else:
                        self._sard_running[t].append([s, None, None, None])

                    m += 1
                    if m >= n:
                        break
                except Empty:
                    continue
            time.sleep(0.0001)

        states = [srds[i][0] for i in range(n)]
        rewards = np.asarray([srds[i][1] for i in range(n)])
        dones = [srds[i][2] for i in range(n)]
        return env_ids, states, rewards, dones

    def get_srd_batch(self, batchsize):
        """Provide (s_{t+1}, r_t, d_t) from different environments.

        Will return 4 list with equal length: env_ids, states, rewards, dones.
        """
        env_ids, states, rewards, dones = self._get_n_srd(batchsize)
        return env_ids, states, rewards, dones

    def put_a_batch(self, env_ids, actions):
        """Return actions to environments.

        env_ids and actions are one-to-one correspondence.
        """
        for env_id, a in zip(env_ids, actions):
            self._action_queue[env_id].put(a)
            if env_id in self._sard_running:
                self._sard_running[env_id][-1][1] = a

    def get_episodes(self, withdraw_running=False):
        """Take out all newly generated episode.

        Args:
            withdraw_running:   If True, clear all running episode queue and
                                withdraw the data.

        Return four list with equal length: s_batch, a_batch, r_batch, d_batch.
        The elements for each list are also list, representing an episode, and
        also are one-to-one correspondence.

        For each episode, the number of states is always 1 greater than dones,
        actions, rewards.
        >>> Examples:
            Element of s_batch: [s0, s1, ..., s_{T-1}, s_T].
            Element of a_batch: [r0, r1, ..., r_{T-1}].
            Element of r_batch: [a0, a1, ..., a_{T-1}].
            Element of d_batch: [d0, d1, ..., d_{T-1}].
        """
        if withdraw_running:
            for env_id in self._sard_running:
                size = len(self._sard_running[env_id])
                half_epi = [self._sard_running[env_id].pop(
                    0) for _ in range(size - 1)]
                half_epi.append(
                    [self._sard_running[env_id][0][0], None, None, None])
                self._sard_finish.append(half_epi)

        s_batch = []
        a_batch = []
        r_batch = []
        d_batch = []
        for episode in self._sard_finish:
            if len(episode) < 4:
                continue
            s_batch.append([sard[0] for sard in episode])
            a_batch.append([sard[1] for sard in episode[:-1]])
            r_batch.append(np.asarray([sard[2] for sard in episode[:-1]]))
            d_batch.append(np.asarray([sard[3] for sard in episode[:-1]]))
        self._sard_finish = []
        return s_batch, a_batch, r_batch, d_batch


    def run(self):
        """Start listening in thread."""
        self._agent_server.serve_forever()


class EnvironmentWrapper:
    """Send srd to the agent and receive actions from the agent."""

    def __init__(self, addr_port_tuple):
        """Connect to agent.

        Args:
            addr_port_tuple:    tuple:(str, int) the address of agent.
        """
        self._addr_port_tuple = addr_port_tuple
        self.env_id = str(uuid4())
        self.action_queue = None
        self.srd_queue = None
        self.reward_function = lambda x: None
        if addr_port_tuple is not None:
            self._prepare_shared_memory_queue()

    def _prepare_shared_memory_queue(self):
        class InferenceMemoryManager(BaseManager):
            pass
        InferenceMemoryManager.register('get_srd')
        InferenceMemoryManager.register('get_a')
        m = InferenceMemoryManager(
            address=self._addr_port_tuple, authkey=b'secret')
        reconnected = 0
        while True:
            try:
                m.connect()
                logging.info('agent connection established')
                break
            except ConnectionRefusedError:
                logging.info('waiting for agent process')
                time.sleep(1)
                reconnected += 1
                if reconnected > 20:
                    logging.warn(
                        f'Warning, env has reconnected {reconnected} times.')

        self.srd_queue = m.get_srd(self.env_id)  # pylint:disable-msg=E1101
        self.action_queue = m.get_a(self.env_id)  # pylint:disable-msg=E1101

    def get_a(self):
        """Retrive action from agent, blocking."""
        a = self.action_queue.get()
        return a

    def put_srd(self, state, reward, done):
        """Put state, reward, done to agent, non-blocking."""
        self.srd_queue.put((state, reward, done))
