import zmq
import msgpack
import numpy as np
from collections import deque

import gym


class RequestClient(object):
    def __init__(self, addr, identity=None):

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)   # pylint: disable=E1101
        if identity is not None:
            self.socket.identity = identity.encode("utf-8")
        self.socket.connect(addr)

    def send(self, content):
        self.socket.send(msgpack.packb(content, use_bin_type=True))

    def recv(self):
        content = self.socket.recv()
        return msgpack.unpackb(content, encoding='utf8')


class RouterReceiver():
    def __init__(self, bind_addr, recv_timeout=-1):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.ROUTER)  # pylint: disable=E1101
        self.socket.set_hwm(1000)
        if recv_timeout > 0:
            self.socket.RCVTIMEO = recv_timeout
        self.socket.bind(bind_addr)

    def recv(self):
        addr, _, content = self.socket.recv_multipart()
        return addr, msgpack.unpackb(content, encoding='utf8')

    def send(self, content, to):
        self.socket.send_multipart(
            [to, b'', msgpack.packb(content, use_bin_type=True)])

    def recv_batch(self, batchsize):
        addrs = []
        contents = []
        for _ in range(batchsize):
            addr, content = self.recv()
            addrs.append(addr)
            contents.append(content)

        return addrs, contents

    def send_batch(self, addrs, contents):
        for (addr, content) in zip(addrs, contents):
            self.send(content, addr)


class SubEnv(object):
    def __init__(self, envs: list):
        """
        :param envs: a list of environment supporting reset, step
        :return: None
        """
        self.envs = envs
        self.obs = list()
        self.actions = list()
        self.rewards = list()
        self.dones = list()

    def reset(self):

        obs = [env.reset() for env in self.envs]
        obs = np.asarray(obs)
        return obs

    def step(self, actions):
        pass


class MasterEnv(object):
    """
    作为中转站，将Agent的action传给environments，将environments的反馈传给Agent。
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, actions):
        pass

    def close(self):
        pass
