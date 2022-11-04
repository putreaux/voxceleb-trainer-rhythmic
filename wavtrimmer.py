import librosa
import os
from pydub import AudioSegment

path = "data/train/"
f = open("data/train_list.txt", "r")
lines = f.readlines()


def trim_wav(file_name):
  dur = librosa.get_duration(filename=file_name)
  print(file_name, dur)
  if dur >= 3 and dur <= 6:
      segment = AudioSegment.from_file(file_name)
      segment = segment[:3000]
      segment.export(file_name, format="wav")
  if dur > 6:
      left = dur*1000
      start = 0
      num = 0
      end = 3000
      while left >= 3000:
        segment = AudioSegment.from_file(file_name)
        segment = segment[start:end]
        segment.export(file_name[:-4]+str(num)+".wav", format="wav")
        start += 3000
        end += 3000
        left -= 3000
        num += 1


for line in lines:
    ind = line.index(" ")
    filepath = line[ind+1:-1]
    full_path = path+filepath
    trim_wav(full_path)
