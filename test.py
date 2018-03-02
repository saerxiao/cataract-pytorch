import argparse
import numpy as np
import math
import torch
import torch.autograd as Variable
import torch.nn as nn
from VideoClipLoader import * 
from model import *
import os

def test(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  T = args.T
  cnn = CnnModel(args.cnn_output_size)
  lstm = nn.LSTM(args.cnn_output_size, args.rnn_hidden_size, args.rnn_layers)
  regressor = Regressor(T * args.rnn_hidden_size)

  ckpt = torch.load(args.ckpt_path)
  cnn.load_state_dict(ckpt['cnn'])
  lstm.load_state_dict(ckpt['lstm'])
  regressor.load_state_dict(ckpt['regressor'])
  
  if args.cuda:
    cnn.cuda()
    lstm.cuda()
    regressor.cuda()
  
  videoLoader = VideoClipLoader(args.video_file_test, args.score_file_test, T) 
  result = np.empty([3, videoLoader.len()])
  for i in range(videoLoader.len()):
    videoFrames, score, videoId = videoLoader.getItem(i)
    print("video: {}, frame: {}".format(videoId, i))
    frames = Variable(videoFrames)
    target = Variable(score)
    if args.cuda:
      frames = frames.cuda()
      target = target.cuda()
    cnnOutput = cnn(frames)
    cnnOutput = cnnOutput.view(cnnOutput.size(0), 1, cnnOutput.size(1))
    rnnOutput, rnnHidden = lstm(cnnOutput)
    output = regressor(rnnOutput)
    result[0][i] = videoId
    result[1][i] = output.data[0]
    result[2][i] = target.data[0]
  np.save("{}/result.npy".format(args.result_dir), result)
  
def main():
    #train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser = argparse.ArgumentParser(description="parser for training arguments")
    train_arg_parser.add_argument("--data_name", type=str)
    train_arg_parser.add_argument("--name", type=str)
    train_arg_parser.add_argument("--which_epoch", type=str, default="latest")
    train_arg_parser.add_argument("--T", type=int, default=10)
    train_arg_parser.add_argument("--cnn_output_size", type=int, default=1024)
    train_arg_parser.add_argument("--rnn_layers", type=int, default=1)
    train_arg_parser.add_argument("--rnn_hidden_size", type=int, default=512)
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--cuda", type=int, default=1,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    args = train_arg_parser.parse_args()
    args.video_file_test = "data/{}/video_test.npy".format(args.data_name)
    args.score_file_test = "data/{}/score_test.npy".format(args.data_name)
    args.result_dir = "results/{}/{}".format(args.name, args.which_epoch)
    if not os.path.exists(args.result_dir):
      os.makedirs(args.result_dir)
    args.ckpt_path = "checkpoints/{}/{}.pth".format(args.name, args.which_epoch)
    test(args)

if __name__ == "__main__":
    main()
