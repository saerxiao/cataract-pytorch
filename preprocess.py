from torchvision import transforms
import numpy as np

dsize = 256
transform = transforms.Compose([
  transforms.Resize(dsize),
  transforms.ToTensor(),
])

def processDir(directory):
  print(directory)
  array = []
  for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
      print(filename)
      img = io.imread(filename)
      dimg = transform(img)
      array.append(dimg)
  return array

rootdir = '/data/cataract/frame'
outputfile = 'data/video_{}.npy'.format(dsize)
allVideos = [] 
for dirname in os.listdir(rootdir):
  a = processDir(dirname)
  if a:
    allVideos.append(a)
nparr = np.array(allVideos)
nparr.save(outputfile, nparr) 
