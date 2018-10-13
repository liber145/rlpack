import tensorflow as tf
import gym
from mpi4py import MPI


env = gym.make("CartPole-v0")
num_steps = 10


s = env.reset()
a = env.action_space.sample()
for i in range(num_steps):

    s, r, d, info = env.step(a)

    recv_data = MPI.COMM_WORLD.gather(s, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        # 处理数据。
        if (i+1) % 5 == 0:
            print("recv_data: {}".format(recv_data))

        actions = [env.action_space.sample() for i in range(MPI.COMM_WORLD.Get_size())]
        print("Process {} generate actions: {}".format(MPI.COMM_WORLD.Get_rank(), actions))
        send_data = actions
    else:
        send_data = 1

    a = MPI.COMM_WORLD.scatter(send_data, root=0)
    print("Iteration {} Process {} get action {}".format(i, MPI.COMM_WORLD.Get_rank(), a))

    if d is True:
        env.reset()
