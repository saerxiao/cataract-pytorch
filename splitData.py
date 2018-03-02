import glob
import numpy as np
import os

rootDir = "/data/cataract/data/movies"
l = glob.glob("{}/*.mp4".format(rootDir))
train = 0.7
val = 0.15

L = len(l)
randomList = np.random.permutation(L)
train_end = int(L * train)
val_end = int(L * (train + val))

def moveData(index, des):
  if not os.path.exists(des):
    os.makedirs(des)

  for i in index:
    os.rename(l[i], "{}/{}".format(des, os.path.basename(l[i])))

moveData(randomList[0 : train_end], "{}/train".format(rootDir))
moveData(randomList[train_end : val_end], "{}/val".format(rootDir))
moveData(randomList[val_end : L], "{}/test".format(rootDir))

  
