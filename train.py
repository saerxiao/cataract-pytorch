import argparse
import numpy as np
import math
import torch
import torch.autograd as Variable
import torch.nn as nn
import torch.optim as optim
from VideoClipLoader import * 
from model import *
import os

def save_ckpt(cnn, lstm, regressor, ckpt_filename, use_cuda):
  cnn.eval()
  lstm.eval()
  regressor.eval()
  if use_cuda:
    cnn.cpu()
    lstm.cpu()
    regressor.cpu()
  model = {'cnn': cnn.state_dict(), 'lstm': lstm.state_dict(), 'regressor': regressor.state_dict()}
  torch.save(model, ckpt_filename)
  if use_cuda:
    cnn.cuda()
    lstm.cuda()
    regressor.cuda()
  cnn.train()
  lstm.train()
  regressor.train()

def train(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  T = args.T
  cnn = CnnModel(args.cnn_output_size)
  lstm = nn.LSTM(args.cnn_output_size, args.rnn_hidden_size, args.rnn_layers)
  regressor = Regressor(T * args.rnn_hidden_size)
  print("-------- cnn --------")
  print(cnn)
  print("-------- lstm -------")
  print(lstm)
  print("-------- regressor -------")
  print(regressor)
  criterion = nn.MSELoss()
  cnn_optimizer = optim.Adam(cnn.parameters(), lr=args.cnn_lr)
  rnn_optimizer = optim.Adam(lstm.parameters(), lr=args.rnn_lr)
  regressor_optimizer = optim.Adam(regressor.parameters(), lr=args.regressor_lr)
  videoLoader = VideoClipLoader(args.video_file_train, args.score_file_train, T)
  videoLoaderValidate = VideoClipLoader(args.video_file_val, args.score_file_val, T)

  if args.cuda:
    cnn.cuda()
    lstm.cuda()
    regressor.cuda()
    
  iters = 0
  train_loss = []
  train_loss_file = "{}/train_loss.txt".format(args.checkpoint_dir)
  val_loss_file = "{}/val_loss.txt".format(args.checkpoint_dir)
  for e in range(args.epochs):
    for i in np.random.permutation(videoLoader.len()):
      videoFrames, score, videoId = videoLoader.getItem(i)
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
      cnnOutput = cnnOutput.view(cnnOutput.size(0), 1, cnnOutput.size(1)) 
      # rnnOutput: T x batch_size x rnn_hidden_size, 
      # rnnHidden[0] is h_T: num_layers x batch x rnn_hidden_size, rnnHidden[1] is c_T: num_layers x batch x rnn_hidden_size
      # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L384
      # e.g. http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
      rnnOutput, rnnHidden = lstm(cnnOutput)
      
      # scalar, TODO: experiment whether to all the hidden state of rnn (rnnOutput) or just the one at the last step (rnnHidden)
      output = regressor(rnnOutput)

      loss = criterion(output, target)
      train_loss.append(loss.data[0])
      
      loss.backward()
      cnn_optimizer.step()
      rnn_optimizer.step()
      regressor_optimizer.step()

      iters = iters + 1
      if iters % args.save_latest_interval == 0:
        ckpt_filename = "{}/latest.pth".format(args.checkpoint_dir)
        save_ckpt(cnn, lstm, regressor, ckpt_filename, args.cuda)

      if iters % args.print_interval == 0:
        print("epoch: {}/{}, iter: {}/{}, err: {}".format(e+1, args.epochs, iters, videoLoader.len(), loss.data[0]))

      if iters % args.train_loss_interval == 0:
        with open(train_loss_file,'a') as f:
          f.write("epoch: {}, iter: {}, err: {}\n".format(e+1, iters, np.mean(train_loss)))
        train_loss = []

      if iters % args.validate_interval == 0:
        total_loss = 0
        val_list = np.random.permutation(videoLoaderValidate.len())[0:100]
        for i in val_list:
          videoFrames, score, videoId = videoLoaderValidate.getItem(i)
          frames = Variable(videoFrames)
          target = Variable(score)
          if args.cuda:
            frames = frames.cuda()
            target = target.cuda()
          cnnOutput = cnn(frames)
          cnnOutput = cnnOutput.view(cnnOutput.size(0), 1, cnnOutput.size(1))
          rnnOutput, rnnHidden = lstm(cnnOutput)
          output = regressor(rnnOutput)
          total_loss = total_loss + criterion(output, target).data[0]
        val_err = total_loss/videoLoaderValidate.len()
        print("validation: epoch: {}, iter: {}, err: {}".format(e+1, iters, val_err))
        with open(val_loss_file, 'a+') as f:
          f.write('epoch: {}, iter: {}, err: {}\n'.format(e+1, iters, val_err))

    ## for every n epochs, save a checkpoint
    if (e + 1) % args.checkpoint_interval == 0:
      ckpt_filename = "{}/epoch_{}.pth".format(args.checkpoint_dir, e+1)
      save_ckpt(cnn, lstm, regressor, ckpt_filename, args.cuda)

def main():
    #train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser = argparse.ArgumentParser(description="parser for training arguments")
    train_arg_parser.add_argument("--data_name", type=str)
    train_arg_parser.add_argument("--name", type=str)
    train_arg_parser.add_argument("--epochs", type=int, default=100)
    train_arg_parser.add_argument("--T", type=int, default=10)
    train_arg_parser.add_argument("--cnn_output_size", type=int, default=1024)
    train_arg_parser.add_argument("--rnn_layers", type=int, default=1)
    train_arg_parser.add_argument("--rnn_hidden_size", type=int, default=512)
    train_arg_parser.add_argument("--cnn_lr", type=float, default=1e-7)
    train_arg_parser.add_argument("--rnn_lr", type=float, default=1e-7)
    train_arg_parser.add_argument("--regressor_lr", type=float, default=1e-7)
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--cuda", type=int, default=1,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--checkpoint_interval", type=int, default=1, help="save checkpoint for every n epochs")
    train_arg_parser.add_argument("--save_latest_interval", type=int, default=500, help="save checkpoint for every n iterations")
    train_arg_parser.add_argument("--train_loss_interval", type=int, default=50)
    train_arg_parser.add_argument("--validate_interval", type=int, default=500)
    train_arg_parser.add_argument("--print_interval", type=int, default=50)
    args = train_arg_parser.parse_args()
    args.video_file_train = "data/{}/video_train.npy".format(args.data_name)
    args.score_file_train = "data/{}/score_train.npy".format(args.data_name)
    args.video_file_val = "data/{}/video_val.npy".format(args.data_name)
    args.score_file_val = "data/{}/score_val.npy".format(args.data_name)
    args.checkpoint_dir = "checkpoints/{}".format(args.name)
    if not os.path.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)
    train(args)

if __name__ == "__main__":
    main()
