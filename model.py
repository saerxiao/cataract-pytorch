import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *

class CnnModel(nn.Module):

  def __init__(self, out_features):
    super(CnnModel, self).__init__()
    # 3 input image channel, 96 output channels, 11x11 square convolution
    # kernel, stride 4
    self.cnn = nn.Sequential(
      conv_relu(3, 96, 11, stride=4),
      nn.MaxPool2d(3, stride=2),
      conv_relu(96, 256, 5, padding=2),
      nn.MaxPool2d(3, stride=2),
      conv_relu(256, 384, 3, padding=1),
      conv_relu(384, 384, 3, padding=1),
      conv_relu(384, 256, 3, padding=1)
    )
    ## TODO: fill in the number of features according to the input
    ## input image 3x256x256, output from cnn is 256x14x14
    ## input image 3x128x128, output from cnn is 256x6x6
    self.fc1 = nn.Linear(50176, out_features)
    #self.fc2 = nn.Linear(4096, 2048)
    #self.fc3 = nn.Linear(2048, 1024)

  def forward(self, x):
    x = self.cnn(x)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    #x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

class Regressor(nn.Module):
  
  def __init__(self, in_features):
    super(Regressor, self).__init__()    
    self.fc1 = nn.Linear(in_features, 1)

  def forward(self, x):
    x = x.permute(1, 0, 2)
    x = x.contiguous().view(x.size(0), -1)
    x = self.fc1(x)
    return x
