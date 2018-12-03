import numpy as np
import random

from gym import Env, spaces
from gym.utils import seeding

class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.reset()

    def reset(self):
        #self.s=np.argmax(self.isd)
        self.s=random.choice(self.isd)
        self.g = random.choice(self.isd)
        return self.s,self.g

    def step(self, a):
        transitions = self.P[self.s][a]
        p, s, r, d= transitions[0]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})