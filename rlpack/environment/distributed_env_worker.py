

from multiprocessing import Process
from multiprocessing.managers import BaseManager

import gym


class DistributedEnvClient(Process):
    """
    start on worker client.
    """

    def __init__(self, hostname='localhost', port=50000):
        super().__init__()
        self.gym_env = gym.make("Breakout-v0")
        self.last_done = False

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

        self.srd_queue.put(self.gym_env.reset())

    def run(self):
        print("37 >>>")
        while True:
            action = self.a_queue.get()

            print("41 >>>")

            if self.last_done:
                ob = self.gym_env.reset()
                reward = 0
                done = True
                info = None
                self.last_done = False
            else:
                ob, reward, done, info = self.gym_env.step(action)
                self.last_done = done

            self.srd_queue.put((ob, reward, done, info))


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
