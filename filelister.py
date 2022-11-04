import os

path = "data/train/"

f = open("data/train_list.txt", "w")
list = os.listdir(path)
for id in list:
    for video_id in os.listdir(path+id):
        for utterance in os.listdir(path+id+"/"+video_id):
            if utterance.endswith("wav") and "trimmed" in utterance:
                f.write(id +" " + id+"/"+video_id+"/"+utterance+"\n")

f.close()
