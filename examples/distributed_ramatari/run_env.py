from utils import EnvironmentWrapper
import args 
from rlpack.environment import make_ramatari


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--env", type=str, default="Pong-ramNoFrameskip-v4")
parser.add_argument("--ip", type=str, default="localhost")
parser.add_argument("--port", type=int, default=50000)
args = parser.parse_args()


env_wrapper = EnvironmentWrapper(addr_port_tuple=(args.ip, args.port))

# env side reset

env = make_ramatari(args.env)


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
