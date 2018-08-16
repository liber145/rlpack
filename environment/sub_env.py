import zmq
import time
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


class GameInfo:
    def __init__(self):
        self.reward = 0.0
        self.real_reward = 0.0
        self.length = 0
        self.real_length = 0

    def update(self, reward, done, info):
        self.reward += reward
        self.real_reward += info.get('real_reward') or reward
        self.length += 1
        self.real_length += 1

        return done and self._get_info(info) or {}

    def clear(self):
        self.reward = 0.0
        self.real_reward = 0.0
        self.length = 0
        self.real_length = 0

    def _get_info(self, info):
        _info = {b'reward': self.reward, b'length': self.length}
        self.reward = 0.0
        self.length = 0
        if 'was_real_done' in info:
            if info['was_real_done']:
                _info[b'real_reward'] = self.real_reward
                _info[b'real_length'] = self.real_length
                self.real_reward = 0.0
                self.real_length = 0

        elif 'real_reward' in info:
            _info[b'real_reward'] = self.real_reward
            self.real_reward = 0.0
            self.real_length = 0

        return _info


def set_seed(env, seed=None, was_real_done=True):
    if not was_real_done:
        return

    seed = seed or int(time.time() * 1000) % 2147483647
    try:
        env.unwrapped.ale.setInt(b'random_seed', seed)
    except AttributeError:
        env.seed(seed)


def sub_env(env_fn, identity, url):
    identity = 'SubAgent-{}'.format(identity)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.identity = identity.encode('utf-8')
    socket.connect(url)

    env = env_fn()
    game_info = GameInfo()

    print('subagent {} start!'.format(identity))
    socket.send(b'ready')

    while True:
        action = socket.recv()
        if action == b'reset':
            set_seed(env)
            game_info.clear()
            state = env.reset()
            socket.send(msgpack.dumps(state))
            continue

        if action == b'close':
            env.close()
            socket.close()
            context.term()
            break

        action = msgpack.loads(action)
        next_state, reward, done, origin_info = env.step(action)
        info = game_info.update(reward, done, origin_info)

        if done:
            set_seed(env, was_real_done=origin_info.get('was_real_done', True))
            next_state = env.reset()

        socket.send(msgpack.dumps((next_state, reward, done, info)))
