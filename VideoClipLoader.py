import torch
import torch.utils.data as data
import numpy as np
from torchvision import transforms

class VideoClipLoader():
  ## allVideos is a nested numpy array, [video1_frames, video2_frames, ...]
  def __init__(self, allVideoFile, scoreFile, T):
    allVideos = np.load(allVideoFile)
    self.allVideos = allVideos
    self.scores = np.load(scoreFile)
    self.T = T
    frameLookup = []
    videoLookup = []
    for i in range(len(allVideos)):
      L = len(allVideos[i])-T+1
      frameLookup.append(range(L))
      videoLookup.append(np.full(L, i, dtype=int))
    self.frameLookup = [item for sublist in frameLookup for item in sublist]
    self.videoLookup = [item for sublist in videoLookup for item in sublist] 

  def getItem(self, idx):
    startFrameId = self.frameLookup[idx] 
    T = self.T
    endFrameId = startFrameId + T
    videoId = self.videoLookup[idx]
    arr = np.asarray(self.allVideos[videoId][startFrameId:endFrameId])
    w, h, c = arr.shape[1], arr.shape[2], arr.shape[3]
    arr = arr.reshape((T*w, h, c))
    transform = transforms.ToTensor()
    tarr = transform(arr)
    tarr = tarr.view(c, T, h, w)
    tarr = tarr.permute(1,0,2,3)
    return tarr, torch.from_numpy(np.array([self.scores[videoId]])).float(), videoId

  def len(self):
    return len(self.frameLookup)
    
    
