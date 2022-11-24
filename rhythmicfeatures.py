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
import audb
import audiofile
import opensmile
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

file_path = "D:\\vox1_dev_wav\\wav\\"
sound_list = []

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


config_str = '''
[componentInstances:cComponentManager]
instance[dataMemory].type=cDataMemory

;;; default source
[componentInstances:cComponentManager]
instance[dataMemory].type=cDataMemory

;;; source

\{\cm[source{?}:include external source]}

;;; main section

[componentInstances:cComponentManager]
instance[framer].type = cFramer
instance[lld].type = cPitchJitter
instance[shs].type = cPitchShs
instance[s_scale].type=cSpecScale
instance[windower].type=cWindower
instance[fft].type=cTransformFFT
instance[magphase].type=cFFTmagphase

[windower:cWindower]
reader.dmLevel=framer
writer.dmLevel=windower
gain=1.0
sigma=0.4

[fft:cTransformFFT]
reader.dmLevel=windower
writer.dmLevel=fft
zeroPadSymmetric = 1

[magphase:cFFTmagphase]
reader.dmLevel=fft
writer.dmLevel=magphase

[s_scale:cSpecScale]
reader.dmLevel=magphase
writer.dmLevel=s_scale
copyInputName = 1
processArrayFields = 0
scale=octave
sourceScale = lin
interpMethod = spline
minF = 25
maxF = -1
nPointsTarget = 0
specSmooth = 1
specEnhance = 1
auditoryWeighting = 1

[shs:cPitchShs]
reader.dmLevel=s_scale
writer.dmLevel=shs
F0raw = 1

[framer:cFramer]
reader.dmLevel = wave
writer.dmLevel = framer
copyInputName = 1
frameMode = fixed
frameSize = 0.032
frameStep = 0.016
frameCenterSpecial = left
noPostEOIprocessing = 1


[componentInstances:cComponentManager]
instance[lld].type = cPitchJitter

[lld:cPitchJitter]

reader.dmLevel = wave
writer.dmLevel = lld
F0reader.dmLevel = shs
F0field = F0raw
jitterLocal = 1
shimmerLocalDB = 1
jitterLocalEnv = 0
jitterDDPEnv = 0
shimmerLocal = 1
shimmerLocalEnv = 0
onlyVoiced = 0
logHNR = 1
inputMaxDelaySec = 0.5

;;; sink

\{\cm[sink{?}:include external sink]}

'''

with open('my.conf', 'w') as fp:
    fp.write(config_str)

smile = opensmile.Smile(feature_set='my.conf',feature_level='lld',     loglevel=2,
    logfile='smile.log')


num_fft = 512
hop_len = 256
sr = 16000
# FUNCTIONS FOR CALCULATING RHYTHMIC FEATURES

# TODO: TRAIN WITH ONE SAMPLE
# BEAT TIMES AS 1D VECTOR OF 0S AND 1S
# dataloader
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



def jitter_shimmer(sound):
    #print("soundi sheippii", sound.shape)
    #print(smile.feature_names)
    jitter = smile.process_signal(sound,sr).jitterLocal
    shimmer = smile.process_signal(sound, sr).shimmerLocalDB
    #mfcc_os = smile.process_signal(sound, sr).mfcc1_sma3
    print("jitter shimmer shape", jitter.shape, shimmer.shape, list(jitter), list(shimmer))
    jitter = np.array(list(jitter))
    shimmer = np.array(list(shimmer))
    mfcc_os = np.array(list(mfcc_os))
    return normalize(jitter), normalize(shimmer)


# CONCATENATING RHYTHMIC FEATURES FOR THE AUDIO FILES AND SAVING THE DATA

display_mode = False

def concat_features(sound):
    rms, ec, frames = rms_ecsg(sound)
    es = energy_spectrum(sound)
    t = tempo(sound, frames)
    mf = mfcc(sound)
    dlt = delta(sound)
    dlt2 = delta_delta(sound)
    #print(mf.shape, dlt.shape, rms.shape)
    #jitter, shimmer = jitter_shimmer(sound)
    c1 = np.concatenate((rms, ec, es, mf, dlt, dlt2, t))
    if (display_mode):
        librosa.display.specshow(c1)
        plt.colorbar()
        plt.title('Concatenation')
        plt.tight_layout()
        plt.show()
    #print(c1.shape)
    return c1
