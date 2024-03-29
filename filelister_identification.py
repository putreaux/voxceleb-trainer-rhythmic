import os
import math
import shutil
path = "wav/"

'''
f = open("data/train_list.txt", "w")
list = os.listdir(path)
for id in list:
    for video_id in os.listdir(path+id):
        for utterance in os.listdir(path+id+"/"+video_id):
            if utterance.endswith("wav") and "trimmed" in utterance:
                f.write(id +" " + id+"/"+video_id+"/"+utterance+"\n")

f.close()


'''
f = open("train_list.txt", "w")
f2 = open("test_list.txt", "w")
list = os.listdir(path)
for id in list:
    video_ids = os.listdir(path+id)
    len_train_ids = math.floor(len(video_ids)*0.8) # first 80 percent (roughly) of videos used for training
    for video_id in video_ids[:len_train_ids]:
        for utterance in os.listdir(path+id+"/"+video_id):
            if utterance.endswith("wav") and "trimmed." in utterance or "trimmed0" in utterance:
                f.write(id +" " + id+"/"+video_id+"/"+utterance+"\n")
    for video_id in video_ids[len_train_ids:]:
        for utterance in os.listdir(path+id+"/"+video_id):
            if utterance.endswith("wav") and "trimmed." in utterance or "trimmed0" in utterance:
                f2.write(id +" " + id+"/"+video_id+"/"+utterance+"\n")



f.close()

# tqdm

f = open("train_list.txt", "r")

lines = f.readlines()
target = "train_data/"
occurences = {}

for line in lines:
    data = line.strip().split()
    id = data[0]
    if id not in occurences.keys():
        occurences[id] = 1
    else:
        occurences[id] += 1

occurences2 = sorted(occurences.items(), key=lambda x:x[1])
print(occurences2)

for fn in occurences.keys():
    if occurences[fn] > 50 and occurences[fn] < 200:
        if os.path.exists(path+fn):
            shutil.move(path+fn, target+fn)
