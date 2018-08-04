from abc import ABCMeta, abstractmethod
from multiprocessing import Process
import traceback
import collections
import zmq
import msgpack
import msgpack_numpy
import gym
from middleware.log import logger


msgpack_numpy.patch()


class Client(Process):
    """Env是Clinet，向Agent发送动作请求。"""
    __metaclass__ = ABCMeta

    def __init__(self, identity, n_step):
        super().__init__()
        self.id = identity
        self.n_step = n_step
        self.cnt = 0

    def run(self):
        context = zmq.Context()
        req1_socket = context.socket(zmq.REQ)   # pylint: disable=E1101
        req2_socket = context.socket(zmq.REQ)   # pylint: disable=E1101
        req1_socket.setsockopt(zmq.IDENTITY, self.id)   # pylint: disable=E1101
        req2_socket.setsockopt(zmq.IDENTITY, self.id)   # pylint: disable=E1101

        self.socket1 = req1_socket
        self.socket2 = req2_socket
        self.cont = context

        req1_socket.connect("tcp://localhost:5550")
        req2_socket.connect("tcp://localhost:5559")

        req2_socket.send(b"Ready.")
        req2_socket.recv()
        req2_socket.send(b"I get awake.")
        self.startsend()

        try:
            while True:

                if self._check() is True:

                    msg = self._get_packet()
                    req1_socket.send(msgpack.packb(
                        msg.__dict__))  # 发送整个游戏过程的大礼包
                    msg = req1_socket.recv()                       # 接受信息，不做任何处理

                    logger.debug("Client({}): {}".format(self.id, self.cnt))

                    req2_socket.recv()
                    req2_socket.send(b"I get awake.")
                    self.startsend(msg)
                    self.cnt = 0

                msg = self._get_packet()
                req1_socket.send(msgpack.packb(msg.__dict__))

                msg = req1_socket.recv()
                msg = msgpack.unpackb(msg)
                self._perform_action(msg)
                self.cnt += 1
        except KeyboardInterrupt:
            logger.info("Keyboad interrupt at env ({})".format(self.id))
        except Exception:
            traceback.print_exc()
        finally:
            logger.info("close env ({})".format(self.id))
            req1_socket.close()
            req2_socket.close()
            context.term()

    def close(self):
        """断开所有连接。"""
        pass
        # print("close env ({})".format(self.id))
        # print("socket1:", self.socket1)
        # self.socket1.close()
        # self.socket2.close()
        # self.cont.term()

    @abstractmethod
    def startsend(self, msg=None):
        """开始新的一轮"""

    @abstractmethod
    def _perform_action(self, action):
        """运行来自Agent的动作action；收集trajectory信息。"""

    @abstractmethod
    def _get_packet(self):
        """整理发送给Agent的消息。"""

    @abstractmethod
    def _check(self):
        """检查是否发送大礼包。"""


class Worker(object):
    """Agent是Worker，处理来自Env发过来的信息，回复动作。"""
    __metaclass__ = ABCMeta

    def __init__(self, n_client=32):
        self.n_client = n_client
        self.addrs = []

    def run(self):
        context = zmq.Context()
        router1_socket = context.socket(zmq.ROUTER)  # pylint: disable=E1101
        router2_socket = context.socket(zmq.ROUTER)  # pylint: disable=E1101

        router1_socket.bind("tcp://*:5550")
        router2_socket.bind("tcp://*:5559")

        for _ in range(self.n_client):
            addr, _, msg = router2_socket.recv_multipart()  # pylint: disable=E0632
            self.addrs.append(addr)
        self.getup = True

        n_received = 0

        try:
            while True:

                if self.getup is True:
                    logger.debug("---------------------------")
                    for addr in self.addrs:
                        router2_socket.send_multipart([addr, b"", b"Get up."])
                    for addr in self.addrs:
                        addr, _, msg = router2_socket.recv_multipart()  # pylint: disable=E0632

                    self.getup = False

                addr, _, msg = router1_socket.recv_multipart()
                msg = msgpack.unpackb(msg)

                action = self._get_action(msg)
                router1_socket.send_multipart(
                    [addr, b"", msgpack.packb(action)])

                # 收集游戏数据。
                self._collect_data(msg)

                if self._check(msg) is True:
                    n_received += 1
                    logger.debug("trajectory: {}".format(
                        len(msg[b"trajectory"])))

                    if n_received == self.n_client:
                        n_received = 0
                        self._update_policy()
                        self.getup = True

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt at agent!")
        except Exception:
            traceback.print_exc()
        finally:
            logger.info("Close agent!")
            self.save_model()
            router1_socket.close()
            router2_socket.close()
            context.term()

    @abstractmethod
    def initialize(self):
        """初始化。设置合适的model。"""

    @abstractmethod
    def _get_action(self, msg):
        """计算处理接受信息的动作；处理是否等待更新。"""

    @abstractmethod
    def _collect_data(self, msg):
        """收集收据。"""

    @abstractmethod
    def _check(self, msg):
        """检查是否收到大礼包。"""

    @abstractmethod
    def _update_policy(self):
        """更新Policy。"""

    @abstractmethod
    def save_model(self):
        """保存模型。"""


if __name__ == "__main__":
    num = 32
    worker = Worker(num)
    worker.run()

    for i in range(num):
        client = Client("{}".format(i).encode("ascii"), 1)
        client.start()
