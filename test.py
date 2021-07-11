import gym 
import pdb


env_names = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v0", 
             # "BipedalWalker-v3", "BipedalWalkerHardcore-v3", 
             "CarRacing-v0", "LunarLander-v2", "LunarLanderContinuous-v2"]

for env_name in env_names:
    env = gym.make(env_name)
    print(f"{env_name}, action_space:{env.action_space}, observation_space:{env.observation_space}")
