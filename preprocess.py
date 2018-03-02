from torchvision import transforms
import numpy as np
import os
from PIL import Image
import re

dsize = 256
transform = transforms.Compose([
  transforms.Scale((dsize,dsize)),
#  transforms.ToTensor(),
])

rootDir = "/data/cataract/data/frames"
desDir = "data/pix256_framerate1"

f = open("/data/cataract/data/scores.txt", "r")
fl = f.readlines()
scoreDic = {}
for line in fl:
  s = line.split(",")
  scoreDic[s[0]] = float(s[1])

def processDir(directory):
  print(directory)
  array = []
  for filename in sorted(os.listdir(directory)):
    if filename.endswith(".png") or filename.endswith(".jpg"):
      #print(filename)
      img = Image.open('{}/{}'.format(directory,filename))
      dimg = transform(img)
      array.append(np.asarray(dimg))
  return array

def process(split):
  allVideos = [] 
  scores = []
  dataDir = "{}/{}".format(rootDir, split)
  for videoname in os.listdir(dataDir):
    movieId = re.findall(r'(\d{3})', videoname)
    print(movieId[0], scoreDic[movieId[0]])
    scores.append(scoreDic[movieId[0]])
    a = processDir('{}/{}'.format(dataDir, videoname))
    if a:
      allVideos.append(a)
  videofile = '{}/video_{}.npy'.format(desDir, split)
  scorefile = '{}/score_{}.npy'.format(desDir, split)
  if not os.path.exists(desDir):
    os.makedirs(desDir)
  np.save(videofile, allVideos)
  np.save(scorefile, scores)

process("train")
process("val")
#process("test")
