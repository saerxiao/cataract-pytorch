import torch
import torch.utils.data as data
import numpy as np

class VideoClipLoader():
  ## allVideos is a nested numpy array, [video1_frames, video2_frames, ...]
  def __init__(self, allVideoFile, scoreFile, T):
    self.allVideos = np.load(allVideoFile)
    self.scores = np.load(scoreFile)
    self.T = T
    frameLookup = []
    videoLookup = []
    for i in len(allVideos):
      L = len(allVideos[i])-T+1
      frameLookup.append(np.range(L))
      videoLookup.append(np.full(L, i, dtype=int))
    self.frameLookup = frameLookup.flatten()
    self.videoLookup = videoLookup.flatten()

  def getItem(self, idx):
    startFrameId = self.frameLookup[idx]
    endFrameId = startFrame + T -1
    videoId = self.videoLookup[idx]
    return self.allVideo[videoId][startFrameId:endFrameId], self.scores[videoId]

  def len(self):
    return len(self.frameLookup)
    
    
