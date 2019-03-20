from utils import EnvironmentWrapper
from rlpack.environment import make_ramatari


env_wrapper = EnvironmentWrapper(
    addr_port_tuple=('localhost', 50000))

# env side reset


env = make_ramatari("Pong-ramNoFrameskip-v4")


state, reward, done = env.reset(), 0, False
env_wrapper.put_srd(state=state, reward=reward, done=done)

# action side

while True:
    if done:
        action = env_wrapper.get_a()  # env will send a fake action
        state, reward, done = env.reset(), 0, False
    else:
        action = env_wrapper.get_a()
        state, reward, done, _ = env.step(action)

    env_wrapper.put_srd(state=state, reward=reward, done=done)
