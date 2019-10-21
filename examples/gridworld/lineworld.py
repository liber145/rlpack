"""
游戏空间是直线排开的若干位置。
动作是前后跳动几个位置。
比如，[] [] [] [] [] [] [] [] [] [] 表示10个位置。
三个动作，分别表示朝前，朝后，不动。 
"""
import numpy as np 

class LineWorld:
    def __init__(self, n_state, n_action):
        self._n_state = n_state 
        self._n_action = n_action 
        self._pos = 0
        self._max_episode_steps = n_state * 2

    def step(self, a):
        half = (self._n_action - 1) / 2
        if 1 <= a <= half:
            self._pos = int(max(self._pos - a, 0))
        elif half+1 <= a <= self._n_action - 1:
            self._pos = int(min(self._n_state-1, self._pos+a-half))
        else:
            assert a == 0, "Unknown action."
        
        r = 0.01 
        if self._pos == self._n_state-1:
            r = 1 

        t = np.zeros(self._n_state) 
        t[self._pos] = 1.0
        return t, r, False, None 
    
    def reset(self):
        self._pos = 0
        t = np.zeros(self._n_state) 
        t[0] = 1.0
        return t


    


