

import signal
import sys
from multiprocessing import Process
from multiprocessing.managers import BaseManager

import gym

from ..common.log import logger


def exit_gracefully(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, exit_gracefully)


class DistributedEnvClient(Process):
    """
    start on worker client.
    """

    def __init__(self, env_name: str, hostname='localhost', port=50000):
        super().__init__()
        self.gym_env = gym.make(env_name)
        self._dim_observation = self.gym_env.observation_space.shape
        self._dim_action = self.gym_env.action_space.shape
        self.last_done = False
        self.trajectory_length = 0
        self.trajectory_reward = 0

        class SharedMemoryManager(BaseManager):
            pass

        SharedMemoryManager.register('get_config')
        SharedMemoryManager.register('get_srd')
        SharedMemoryManager.register('get_a')

        self.m = SharedMemoryManager(address=(hostname, port), authkey=b'abab')
        self.m.connect()
        config_queue = self.m.get_config()
        config_json = config_queue.get()
        self.env_id = config_json['env_id']
        self.srd_queue = self.m.get_srd(self.env_id)
        self.a_queue = self.m.get_a(self.env_id)

        s = self.gym_env.reset()
        # logger.info(f"s: {s.shape}  {type(s)}")
        self.srd_queue.put([s])

    def run(self):
        while True:
            action = self.a_queue.get()
            epinfo = {}

            if self.last_done:
                ob = self.gym_env.reset()
                reward = 0
                done = True
                info = None
                self.last_done = False
                epinfo = {"episode": {"r": self.trajectory_reward, "l": self.trajectory_length}}
                self.trajectory_length = 0
                self.trajectory_reward = 0
            else:
                ob, reward, done, info = self.gym_env.step(action)
                self.last_done = done
                self.trajectory_length += 1
                self.trajectory_reward += reward

            self.srd_queue.put((ob, reward, done, epinfo))

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action


if __name__ == '__main__':
    n_env = 8
    processes = []
    for _ in range(n_env):
        p = DistributedEnvClient()
        p.daemon = True
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
