#Settings things up
import numpy as np
#import sounddevice as sd
import scipy.io.wavfile as wf
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import soundfile as sf
import io
import os
from os.path import join


from derivative import dxdt
import librosa                     # librosa music package


import IPython                     # for playing audio
import numpy as np                 # for handling large arrays
import pandas as pd                # for data manipulation and analysis
import scipy                       # for common math functions
import sklearn                     # a machine learning library
import os                          # for accessing local files

import librosa.display             # librosa plot functions
import matplotlib.pyplot as plt    # plotting with Matlab functionality
import seaborn as sns              # data visualization based on matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import soundfile as sf

#file_path = "D:\\vox1_dev_wav\\wav\\"
#sound_list = []

'''f = open("train_list_full_trimmed.txt", "r")
lines = f.readlines()

for line in lines[:1000]:
    ind = line.index(" ")
    filepath = line[ind+1:-1]
    sound_list.append(filepath)

'''
def normalize(arr):
    normalized = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return normalized

num_fft = 512
hop_len = 256
sr = 16000

# FUNCTIONS FOR CALCULATING RHYTHMIC FEATURES

#root mean square and energy contour spectrogram
def rms_ecsg(sound):
    #sound , sr = librosa.load(file_path+sound, sr=16000)
    S, phase = librosa.magphase(librosa.stft(sound, n_fft=num_fft, hop_length=hop_len) )# compute magnitude and phase content

    rms = librosa.feature.rms(S=S, frame_length=num_fft, hop_length=hop_len) # compute root-mean-square for each frame in magnitude
    ecsg = librosa.amplitude_to_db(S, ref=np.max)
    #print("rms shape: ", rms.shape, "energy contour spectrogram shape: ", ecsg.shape)
    framenum = rms.shape[1]
    '''plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.semilogy(rms.T, label='RMS Energy', color="black")
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend()
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    plt.title('log power energy contour spectrogram for sound ' + sound)
    '''
    return normalize(rms), normalize(ecsg), framenum

# using an energy (magnitude) spectrum
def energy_spectrum(sound):
    #sound , sr = librosa.load(file_path+sound, sr=16000)
    S = np.abs(librosa.stft(sound, n_fft=num_fft, hop_length=hop_len)) # apply short-time fourier transform
    chroma_e = librosa.feature.chroma_stft(S=S, sr=sr)
    #print("energy spectrum chromagram shape: ", chroma_e.shape)
    '''plt.figure(figsize=(15, 4))
    librosa.display.specshow(chroma_e, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Energy spectrum chromagram for sound file ' + sound)
    plt.tight_layout()
    plt.show()
    '''
    return normalize(chroma_e)

def tempo(sound, frames):
    #sound , sr = librosa.load(file_path+sound, sr=16000)
    y_harmonic, y_percussive = librosa.effects.hpss(sound)
    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)

    #print('Detected Tempo: ' + str(tempo) + ' speech unit/min')
    beat_times = list(librosa.frames_to_time(beat_frames, sr=sr))
    beat_time_diff = np.ediff1d(beat_times)
    beat_nums = np.arange(1, np.size(beat_times))
    #print("beat times shape: ", beat_times.shape, "beat nums shape: ", beat_nums.shape)
    tempo_array = np.zeros(frames)
    #print(beat_frames)
    tempo_array[beat_frames] = 1
    #print(tempo_array.shape, tempo_array)
    return tempo_array.reshape(1, len(tempo_array))
    '''fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    ax.set_ylabel("Time difference (s)")
    ax.set_xlabel("speech units ")
    g = sns.barplot(beat_nums, beat_time_diff, palette="rocket",ax=ax)
    g = g.set(xticklabels=[])

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    ax.set_ylabel("Time difference (s)")
    ax.set_xlabel("speech units ")
    '''

def mfcc(sound):
    #sound , sr = librosa.load(file_path+sound, sr=16000)
    stft = librosa.stft(sound, n_fft=num_fft, hop_length=hop_len)
    mfcc = librosa.feature.mfcc(S=stft, sr=sr, n_fft=num_fft, hop_length=hop_len)
    #print("mfcc shape: ", mfcc.shape)
    return normalize(mfcc)

def delta(sound):
    mfcc_sound = mfcc(sound)
    d1 = librosa.feature.delta(mfcc_sound)
    return normalize(d1)


def delta_delta(sound):
    mfcc_sound = mfcc(sound)
    d2 = librosa.feature.delta(mfcc_sound, order=2)
    return normalize(d2)


# CONCATENATING RHYTHMIC FEATURES FOR THE AUDIO FILES AND SAVING THE DATA

display_mode = False

def concat_features(sound):
    rms, ec, frames = rms_ecsg(sound)
    es = energy_spectrum(sound)
    t = tempo(sound, frames)
    mf = mfcc(sound)
    dlt = delta(sound)
    dlt2 = delta_delta(sound)
    c1 = np.concatenate((rms, ec, es, mf, dlt, dlt2, t))
    if (display_mode):
        librosa.display.specshow(c1)
        plt.colorbar()
        plt.title('Concatenation')
        plt.tight_layout()
        plt.show()
    #print(c1.shape)
    return c1