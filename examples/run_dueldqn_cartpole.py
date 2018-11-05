from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import deque 

from rlpack.common import Memory
from rlpack.environment import CartpoleWrapper
from rlpack.algos import DuelDQN

class Config(object):
    self.n_env = 4
    self.dim_observation = (4,) 
    self.n_action = 2

    # 训练长度和周期
    self.warm_start = 100
    self.trajectory_length = 2
    self.update_step = 100000
    self.update_freq = 1

    # 训练参数
    self.batch_size = 32
    self.memory_size = 10000
    self.discount = 0.99
    self.lr_schedule = lambda x: (1-x) * 2.5e-4
    self.epsilon_schedule = lambda x: (1-x) * 0.9
    self.lr = 1e-3
    self.update_target_freq = 100

    self.seed = 1
    self.save_path = "./log/cartpole_2"
    self.save_model_freq = 0.001
    self.log_freq = 1000


def learn(env, agent, config):
    memory = Memory(config.memory_size)
    epinfobuf = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    obs = env.reset()
    for i in tqdm(range(config.warm_start)):
        actions = agent.get_action(obs)
        next_obs, rewards, dones, infos = env.step(actions)
        memory.store_sard(obs, actions, rewards, dones)
        obs = next_obs

    for i in tqdm(config.update_step):
        epinfos = []
        for _ in range(config.trajectory_length):
            actions = agent.get_action(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            memory.store_sard(obs, actions, rewards, dones)
            obs = next_obs

            for info in infos:
                maybeepinfo = info.get("episode")
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

        epinfobuf.extend(epinfos)
        summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
        summary_writer.add_scalar("eplenmean", safemean([epinfo["l"] for epinfo in epinfobuf]), global_step=i)

        if i % config.update_freq == 0:
            data_batch = memory.sample_transition(config.batch_size)
            agent.update(data_batch, i / config.update_step)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo["l"] for epinfo in epinfobuf])
            tqdm.write(f"eprewmean: {rewmean} eplenmean: {lenmean}")

if __name__ == "__main__":
    config = Config()
    env = CartpoleWrapper(4)
    agent = DuelDQN
    learn(env, agent, config)
