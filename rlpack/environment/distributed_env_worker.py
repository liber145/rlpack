import signal
import sys
from multiprocessing import Process
from multiprocessing.managers import BaseManager


def exit_gracefully(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, exit_gracefully)


class DistributedEnvClient(Process):
    """
    Start on worker client.
    """

    def __init__(self, env, hostname='localhost', port=50000):
        super().__init__()
        self.env = env
        self._dim_observation = self.env.dim_observation
        self._dim_action = self.env.dim_action

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

        s = self.env.reset()
        self.srd_queue.put([s])

    def run(self):
        """Run forever. If done, reset."""
        while True:
            action = self.a_queue.get()
            info = {}

            ob, reward, done, info = self.env.step(action)

            if done:
                ob = self.env.reset()

            self.srd_queue.put((ob, reward, done, info))

    @property
    def dim_observation(self):
        """The dimension of observatin."""
        return self._dim_observation

    @property
    def dim_action(self):
        """The dimension of action.

        For discrete-action game, it means the number of actions.
        For continuous-action game, it means the dimension of action.
        """
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
