#Settings things up
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
import soundfile
import opensmile
from os.path import join
import parselmouth
import math
import time
import random
import glob
from scipy import signal

from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import soundfile as sf

file_path = "D:\\vox1_dev_wav\\wav\\"
sound_list = []

train_path = "data/train"
train_list_path = "data/train_list.txt"
test_list_path = "data/test_list.txt"
feature_save_path = "data/train_features/"
musan_path = "data/musan_split"
rir_path = "data/RIRS_NOISES/simulated_rirs"

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

def mfcc(sound):
    #sound , sr = librosa.load(file_path+sound, sr=16000)
    stft = np.abs(librosa.stft(sound, n_fft=num_fft, hop_length=hop_len))
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

def mel_spectrogram(sound):
    mels = librosa.feature.melspectrogram(sound, sr=sr)
    return normalize(mels)


def jitter_shimmer(sound, f0min, f0max):
    audio_praat = parselmouth.Sound(values=sound, sampling_frequency=16000)
    #print(audio_praat.sampling_period, audio_praat.n_samples)
    pointProcess = call(audio_praat, "To PointProcess (periodic, cc)", f0min, f0max)
    jitters = []
    shimmers = []
    for time in range(-hop_len, sound.shape[0]-hop_len, hop_len):
        point_to_sec = max(0,time/16000)
        point2 = min(sound.shape[0],(time+num_fft)/16000)
        jitter = call(pointProcess, "Get jitter (local)", point_to_sec, point2, 0.0001, 0.02, 3)
        shimmer = call([audio_praat, pointProcess], "Get shimmer (local_dB)", point_to_sec, point2, 0.0001, 0.02, 1.3, 3.6)
        jitters.append(jitter)
        shimmers.append(shimmer)
    jitters = np.nan_to_num(np.array(jitters))
    shimmers = np.nan_to_num(np.array(shimmers))
    return normalize(jitters).reshape(1, jitters.shape[0]), normalize(shimmers).reshape(1, shimmers.shape[0])


# CONCATENATING RHYTHMIC FEATURES FOR THE AUDIO FILES AND SAVING THE DATA

display_mode = False

def concat_features(sound):
    rms, ec, frames = rms_ecsg(sound)
    es = energy_spectrum(sound)
    start = time.time()
    t = tempo(sound, frames)
    end = time.time()
    mf = mfcc(sound)
    dlt = delta(sound)
    dlt2 = delta_delta(sound)
    jitter, shimmer = jitter_shimmer(sound, 50, 200)
    #print(id, ": fe duration: ",  end - start)
    c1 = np.concatenate((rms, ec, es, mf, dlt, dlt2, t, np.nan_to_num(jitter), np.nan_to_num(shimmer)))
    return np.around(c1, 6)


def loadWAV(filename, max_frames):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = librosa.load(filename, sr=16000)



    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]
    startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    for asf in startframe:
        feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats,axis=0).astype(float)

    return feat

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4)

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio
    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)

        rir, fs     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

def save_features(sound_list, trainfeats):
    with open(sound_list) as dataset_file:
        lines = dataset_file.readlines()
    # Make a dictionary of ID names and ID indices
    dictkeys = list(set([x.split()[0] for x in lines]))
    dictkeys.sort()
    dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
    print(dictkeys)
    augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=200)

    # Parse the training list into file names and ID indices
    for lidx, line in enumerate(lines):
        data = line.strip().split();
        speaker_label = data[0]
        filename = os.path.join(train_path,data[1])
       # print(os.path.join(feature_save_path, data[1][:-4]), os.path.exists(os.path.join(feature_save_path, data[1][:-4])))
       # a = 2 / 0
        index_of_file = data[1].find("/", data[1].find("/") + 1) + 1
        newdir = feature_save_path + data[1][:index_of_file]
        if os.path.isdir(train_path+"/"+data[0]):
            audio = loadWAV(filename, 200)
            augtype = 0
            if trainfeats:
                augtype = random.randint(1, 4)
            if augtype == 1:
                audio = augment_wav.reverberate(audio)
            elif augtype == 2:
                audio = augment_wav.additive_noise('music', audio)
            elif augtype == 3:
                audio = augment_wav.additive_noise('speech', audio)
            elif augtype == 4:
                audio = augment_wav.additive_noise('noise', audio)
            features = concat_features(audio.flatten())
            index_of_file = data[1].find("/", data[1].find("/")+1)+1
            newdir = feature_save_path+data[1][:index_of_file]
            os.makedirs(newdir, exist_ok=True)
            if lidx % 10 == 0:
                print("Extracting features: {}%".format(str(100.00*(lidx/len(lines)))))
            np.savetxt(newdir+data[1][index_of_file:-4], features, fmt='%.6f')

#save_features(train_list_path, True)
#save_features(test_list_path, False)