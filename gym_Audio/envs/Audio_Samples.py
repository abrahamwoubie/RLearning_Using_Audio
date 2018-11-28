import matplotlib.pyplot as plt # For plotting
import numpy as np # to work with numerical data efficiently
from scipy.spatial import  distance
import pylab

fs=100

def generate():
    return pylab.specgram(samples, Fs=fs)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('spectrogram1.png')

if __name__ == '__main__':

    fs = 100 # sample rate
    f = 2 # the frequency of the signal

    x = np.arange(fs) # the points on the x axis for plotting

    # compute the value (amplitude) of the sin wave at the for each sample
    samples = [ np.sin(2*np.pi*f * (i/fs)) for i in x]


    d=pylab.specgram(samples,Fs=fs)

