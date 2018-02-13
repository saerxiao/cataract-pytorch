import numpy as np
import math
import torch

import torch
import torch.autograd.Variable as Variable
import torch.nn as nn
import torch.optim as optim
import VideoClipLoader
from model import *


def train(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  T = args.T
  cnn = CnnModel(args.cnn_output_size)
  rnn_layers = arges.rnn_layers if args.rnn_layers else 1
  lstm = nn.LSTM(args.cnn_output_size, args.rnn_hidden_size, rnn_layers)
  regressor = Regressor(args.rnn_hidden_size)
  criterion = nn.MSELoss()
  cnn_optimizer = optim.Adam(cnn.parameters(), lr=args.cnn_lr)
  rnn_optimizer = optim.Adam(lstm.parameters(), lr=args.rnn_lr)
  regressor_optimizer = optim.Adam(regressor.parameters(), lr=args.regressor_learning_rate)
  videoLoader = VideoClipLoader(args.video_file_train, args.score_file_train, T)
  videoLoaderValidate = VideoClipLoader(args.video_file_val, args.score_file_val, T)

  if args.cuda:
    cnn.cuda()
    lstm.cuda()
    regressor.cuda()
    
  iters = 0
  train_loss = []
  for e in range(args.epochs):
    for i in np.range(videoLoader.len()):
      videoFrames, score = videoLoader.getItem(i)
      # videoFrames is TxCxHxW
      frames = Variable(videoFrames)
      target = Variable(score)
      if args.cuda:
        frames = frames.cuda()
        target = target.cuda()

      # T x cnn_output_size
      cnnOutput = cnn(frames)
      # TODO: cnn_batch_size is tuned to make the CNN part fit into memory
      #cnnB = args.cnn_batch_size
      #cnnOutput = torch.Tensor(T, args.cnn_output_size)
      #kstart = 0
      #for k in range(int(math.ceil(float(T)/cnnB))):
      #  kend = (k+1) * cnnB
      #  if kend > T:
      #    kend = T
      #  frames_k = Variable(videoFrames[(kstart, kend),:,:,:])
      #  cnnOutput_k = cnnModel(frames_k)
      #  cnnOutput[(kstart,kend),:] = cnnOutput_k
      #  kstart = kend + 1

      # reshape cnnOutput to T x batch_size x cnn_output_size to satisfy the input size requirement for RNN/LSTM, set batch_size = 1
      cnnOutput = cnnOuput.view(cnnOutput.size(1), 1, cnnOutput.size(2)) 
      # rnnOutput: T x batch_size x rnn_hidden_size, rnnHidden: num_layers x batch x rnn_hidden_size
      # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L384
      # e.g. http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
      rnnOutput, rnnHidden = lstm(cnnOutput)
      
      # scalar, TODO: experiment whether to all the hidden state of rnn (rnnOutput) or just the one at the last step (rnnHidden)
      output = regressor(rnnOutput)

      loss = criterion(output, target)
      train_loss.append(loss)
      
      loss.backward()
      cnn_optimizer.step()
      rnn_optimizer.step()
      regressor_optimizer.step()

      iters = iters + 1
      if iters % args.train_loss_interval == 0:
        with open(args.train_loss_file,'a') as f:
          np.savetxt(f, train_loss)
        train_loss = []

      if iters % args.validate_interval == 0:
        total_loss = 0
        for i in np.range(videoLoaderValidate.len()):
          videoFrames, score = videoLoaderValidate.getItem(i)
          frames = Variable(videoFrames)
          target = Variable(score)
          if args.cuda:
            frames = frames.cuda()
            target = target.cuda()
          cnnOutput = cnn(frames)
          cnnOutput = cnnOuput.view(cnnOutput.size(1), 1, cnnOutput.size(2))
          rnnOutput = lstm(cnnOutput)
          output = regressor(rnnOutput)
          total_loss = total_loss + criterion(output, target)
        with open(args.validate_loss_file, 'a+') as f:
          f.write('{} {}\n'.format(iters, total_loss/videoLoaderValidate.len())

    ## for every n epochs, save a checkpoint
    if args.checkpoint_model_dir is not None and (e + 1) % args.checkpoint_interval == 0:
      cnn.eval()
      lstm.eval()
      regressor.eval()
      if args.cuda:
        cnn.cpu()
        lstm.cpu()
        regressor.cpu()
      ckpt_model_filename = "ckpt_epoch_" + str((e+1)*nEpoch) + ".pth"
      ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
      model = {'cnn': cnn.state_dict(), 'lstm': lstn.state_dict(), 'regressor': regressor.state_dict()}
      torch.save(model, ckpt_model_path)
      if args.cuda:
        cnn.cuda()
        lstm.cuda()
        regressor.cuda()
      cnn.train()
      lstm.train()
      regressor.train()

def main():
    #train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser = argparse.ArgumentParser(description="parser for training arguments")
    train_arg_parser.add_argument("--video_file_train", type=str, default='file path')
    train_arg_parser.add_argument("--score_file_train", type=str, default='file path')
    train_arg_parser.add_argument("--video_file_val", type=str, default='file path')
    train_arg_parser.add_argument("--score_file_val", type=str, default='file path')
    train_arg_parser.add_argument("--T", type=int, default=10)
    train_arg_parser.add_argument("--cnn_output_size", type=int, default=1024)
    train_arg_parser.add_argument("--rnn_hidden_size", type=int, default=512)
    train_arg_parser.add_argument("--cnn_lr", type=float, default=1e-4)
    train_arg_parser.add_argument("--rnn_lr", type=float, default=1e-4)
    train_arg_parser.add_argument("--regressor_lr", type=float, defult=1e-4)
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--cuda", type=int, default=1,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--train_loss_file", type=str, default='file path')
    train_arg_parser.add_argument("--train_loss_interval", type=int, default=50)
    train_arg_parser.add_argument("--validate_loss_file", type=str, default='file path')
    train_arg_parser.add_argument("--validate_interval", type=int, default=100)
    args = train_arg_parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
