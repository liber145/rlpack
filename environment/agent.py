import time
import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import os
from middleware.mq import Worker
from middleware.memory import Memory
from estimator import DQN, SoftDQN, DoubleDQN, AveDQN, DistDQN, PG, TRPO, A2C, PPO, PPO, DDPG
from middleware.log import logger
from environment.scaler import Scaler


class Agent(Worker):
    def __init__(self, base_path, model_name, lr, n_client, discount, batch_size, memory_size, n_act, dim_ob):
        super().__init__(n_client)
        self.model_path = os.path.join(base_path, "models")
        self.event_path = os.path.join(base_path, "events")
        if not os.path.exists(self.event_path):
            os.makedirs(self.event_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model_name = model_name
        self.lr = lr
        self.batch_size = batch_size
        self.n_act = n_act
        self.dim_act = int(n_act)
        self.dim_ob = dim_ob
        self.discount = discount

        self.starttime = time.time()

        self.estimator = self._get_estimator()
        self.buffer = Memory(memory_size)

        self.summary_writter = SummaryWriter(self.event_path)

        self.eprewards = []
        self.cnt = 1

    def _get_estimator(self):
        if self.model_name == "dqn":
            model = DQN(self.dim_ob, self.n_act, self.lr, self.discount)
        elif self.model_name == "softdqn":
            model = SoftDQN(self.dim_ob, self.n_act, self.lr, self.discount)
        elif self.model_name == "doubledqn":
            model = DoubleDQN(self.dim_ob, self.n_act, self.lr, self.discount)
        elif self.model_name == "avedqn":
            model = AveDQN(self.dim_ob, self.n_act, self.lr, self.discount, 2)
        elif self.model_name == "distdqn":
            model = DistDQN(self.dim_ob, self.n_act, self.lr, self.discount)
        elif self.model_name == "pg":
            model = PG(self.dim_ob, self.dim_act, self.lr, self.discount)
        elif self.model_name == "trpo":
            model = TRPO(self.dim_ob, self.dim_act, self.lr, self.discount)
        elif self.model_name == "ppo":
            model = PPO(self.dim_ob, self.dim_act, self.lr, self.discount)
        elif self.model_name == "a2c":
            model = A2C(self.dim_ob, self.dim_act, self.lr, self.discount)
        elif self.model_name == "ddpg":
            model = DDPG(self.dim_ob, self.dim_act, self.lr, self.discount)
        else:
            logger.error("Unrecognized model name!")

        model.load_model(self.model_path)
        print("Use model: {}".format(self.model_name))
        return model

    def _get_action(self, msg):
        obs = msg[b"state"]
        if obs.ndim == 3:   # 图片
            obs = obs[np.newaxis, :]
        if obs.ndim == 1:   # 向量特征
            obs = obs[np.newaxis, :]

        actions = self.estimator.get_action(obs, 0.1)
        return actions[0]

    def _collect_data(self, msg):
        if msg[b"trajectory"] is not None:
            self.buffer.append(msg[b"trajectory"])

        # 每次收到数据后，对数据进行scale操作。然后存入memory中。

        self.recv_episode_reward = msg[b"episode_reward"]
        self.recv_id = msg[b"id"]
        self.recv_nstep = msg[b"nstep"]

        if msg[b"episode_reward"] is not None:
            # logger.info("episode_reward: {}".format(msg[b"episode_reward"]))
            self.eprewards.append(msg[b"episode_reward"])

    def _check(self, msg):
        if msg[b"trajectory"] is None:
            return False
        else:
            return True

    def _update_policy(self):

        if len(self.buffer.mem) >= self.batch_size:
            logger.debug("buffer size: {}".format(len(self.buffer.mem)))
            logger.debug("batch size: {}".format(self.batch_size))

            total_t, result = self.estimator.update(
                self.buffer.sample(self.batch_size))

            # 每次使用最新policy采样的样本。
            self.buffer.clear()

            self.summary_writter.add_scalar("loss", result["loss"], total_t)

            if self.recv_episode_reward is not None:
                self.summary_writter.add_scalar("episode_reward/id_{}".format(
                    self.recv_id.decode("ascii")), self.recv_episode_reward, self.recv_nstep)

            if total_t >= self.cnt * 100:
                self.cnt += 1
                logger.info("step: {}\t Loss: {}".format(
                    total_t, result["loss"]))
                logger.info("mean ep reward: {}".format(
                    np.mean(self.eprewards)))
                self.eprewards = []

            if total_t % 1000 == 0:
                dtime = time.time() - self.starttime
                print("Save model, global_step: {}, delta time: {}.".format(
                    total_t, dtime))
                self.save_model()
                self.starttime = time.time()

    def save_model(self):
        self.estimator.save_model(self.model_path)
