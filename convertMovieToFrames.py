import glob
import subprocess
import os

dataDir = "/data/cataract/data/movies"
frameDir = "/data/cataract/data/frames"

def convert(split):
  movies = glob.glob("{}/{}/*.mp4".format(dataDir, split))
  for m in movies:
    print(m)
    basename = os.path.splitext(os.path.basename(m))[0]
    desDir = "{}/{}/{}".format(frameDir, split, basename) 
    if not os.path.exists(desDir):
      os.makedirs(desDir) 
    subprocess.run(["avconv", "-i", m, "-r", "1", "-f", "image2", "{}/%08d.png".format(desDir)]) 

convert("train")
convert("val")
#convert("test")
