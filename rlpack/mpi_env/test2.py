from main import Environment
from mpi4py import MPI


class Policy(object):
    def __init__(self):
        pass

    def get_action(self, feedback):
        if MPI.COMM_WORLD.Get_rank() == 0:
            num_action = len(feedback)
            return []


if __name__ == "__main__":

    env = Environment()
    s = env.reset()
    a = [env.env.action_space.sample() for i in range(4)]

    for i in range(1, 11):
        s_batch = env.step(a)

        if i % 5 == 0:
            print("Process {}: s: {}".format(MPI.COMM_WORLD.Get_rank(), s_batch))

        a = [env.env.action_space.sample() for i in range(4)]
