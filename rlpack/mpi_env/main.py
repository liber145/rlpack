from mpi4py import MPI
import numpy as np
import gym

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()


# env = gym.make("CartPole-v1")
# s = env.reset()
#
#
# send_data = s
# print(f"Process {rank} send data {send_data} to root")
# recv_data = comm.gather(send_data, root=0)
#
# if rank == 0:
#     print(f"Process {rank} gather all data {recv_data}")

# 调用主进程。
def master_only(func):
    def _call_master(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            return func(*args, **kwargs)
    return _call_master


class Policy(object):
    @master_only
    def __init__(self):
        pass

    @staticmethod
    @master_only
    def get_action(state):
        return [np.random.randint(2) for _ in range(4)]


class Environment(object):
    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def reset(self):
        s = self.env.reset()
        recv_data = MPI.COMM_WORLD.gather(s, root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            return recv_data

    def step(self, action_list):
        a = MPI.COMM_WORLD.scatter(action_list, root=0)
        feedback = self.env.step(a)

        recv_data = MPI.COMM_WORLD.gather(feedback, root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            return recv_data


if __name__ == "__main__":

    pol = Policy()
    env = Environment()
    s = env.reset()

    for i in range(1, 11):
        a = pol.get_action(s)
        s_batch = env.step(a)

        if MPI.COMM_WORLD.Get_rank() == 0:
            if i % 5 == 0:
                print("Process {}: s: {}".format(MPI.COMM_WORLD.Get_rank(), s_batch))
