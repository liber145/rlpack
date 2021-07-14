import random
from collections import deque

class ReplayBuffer:
    def __init__(self, size):
        self.size = 0
        self.max_size = size
        self.buffer = deque()

    def append(self, s, a, r, ns, done, info):
        if self.size<self.max_size:
            self.buffer.append((s, a, r, ns, done, info))
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s, a, r, ns, done, info))

    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        return batch