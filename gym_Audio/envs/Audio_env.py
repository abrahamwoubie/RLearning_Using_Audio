import sys
import sklearn
from six import StringIO
from gym import utils
import aubio
from gym_Audio.envs import discrete
import numpy as np
from scipy.spatial import distance

import pandas as pd # data frame
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt # to view graphs

# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing


dist={}
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

goal_row=0
goal_col=0

Features_of_Goal={}
Features_of_Current_State={}

MAPS = {
    "Grid": [
        "SCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCG"
    ],
}

'''
MAPS = {
    "10x10": [
        "SCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCC",
        "CCCCCCCCCG"
    ],
}
'''
class AudioEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="Grid"):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol


        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        PP={}
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        def Extract_Pitch(row, col):

            pitch_List = []
            sample_rate = 44100
            x = np.zeros(44100)
            for i in range(44100):
                x[i] = np.sin(2. * np.pi * i * 225. / sample_rate)

            # create pitch object
            p = aubio.pitch("yin", samplerate=sample_rate)

            # pad end of input vector with zeros
            pad_length = p.hop_size - x.shape[0] % p.hop_size
            x_padded = np.pad(x, (0, pad_length), 'constant', constant_values=0)
            # to reshape it in blocks of hop_size
            x_padded = x_padded.reshape(-1, p.hop_size)

            # input array should be of type aubio.float_type (defaults to float32)
            x_padded = x_padded.astype(aubio.float_type)

            for frame, i in zip(x_padded, range(len(x_padded))):
                time_str = "%.2f" % (i * p.hop_size / float(sample_rate))
                pitch_candidate = p(frame)[0] + row + col
                # print(pitch_candidate)
                pitch_List.append(pitch_candidate)
            return pitch_List


        for row in range(0, nrow):
            for col in range(0, ncol):
                letter = desc[row, col]
                if letter in b'G':
                    goal_pitch_values = Extract_Pitch(row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                Current_Pitch_values = Extract_Pitch(row, col)
                dist[row, col] = distance.euclidean(goal_pitch_values,Current_Pitch_values)
                for a in range(4):
                    li = P[s][a]
                    if dist[row, col]==0:
                        li.append((1.0, s, 1, True))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        done = False
                        rew = 0
                        li.append((1.0, newstate, rew, done))
        super(AudioEnv, self).__init__(nS, nA, P, isd)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile



