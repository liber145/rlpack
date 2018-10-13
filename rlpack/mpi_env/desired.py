import numpy as np

num_step = 10000


class Environment(object):
    def __init__(self):
        """初始化游戏。"""
        pass

    def reset(self):
        """重置游戏。返回重置后的状态。"""
        state = np.random.rand(12, 12, 3)
        return state

    def step(self, action):
        """执行action，返回(状态，奖励，是否结束)。"""
        state = np.random.rand(12, 12, 3)
        reward = np.random.rand(1)
        done = np.random.rand() > 0.5
        return state, reward, done


class Policy(object):
    def __init__(self):
        """初始化策略，超参数配置，网络搭建等"""
        pass

    def get_action(self, state):
        """根据state，返回对应动作，可以是确定性，也可以是不确定性"""
        pass

    def update_policy(self, minibatch):
        """更新策略。"""
        pass


class Memory(object):
    """存储中间数据。"""

    def __init__(self):
        """初始化memory。"""
        pass

    def put_data(self, data):
        """存入数据。"""
        pass

    def sample(self):
        """抽取minibatch的数据。"""
        pass



# 训练过程。
env = Environment()
pol = Policy()
mem = Memory()

train_freq = 100
num_train_step = 10000


feedback = env.reset()
action = pol.get_action(feedback)
for i in range(num_train_step):

    feedback = env.step(action)
    mem.put_data(feedback)

    if i % train_freq == 0:
        minibatch = mem.sample()
        pol.update_policy(minibatch)

    if feedback.done is True:
        feedback = env.reset()

    action = pol.get_action(feedback)
