from decimal import DivisionImpossible
from tkinter import E
from typing import Optional, Union
from numpy import size
import torch
import torch.nn as nn
import torch.distributions as D
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

from .functions import *
from .common import *

class NewCNN(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ReLU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (3, 3, 4, 4)
        stride = 2
        padding = 1
        d = 64

        self.hist_size = 2

        self.model = nn.Sequential(
            nn.Conv2d(self.hist_size * 3, d, 4, stride, bias=False),
            nn.BatchNorm2d(d),
            activation(),
            nn.ConvTranspose2d(d, in_channels, 4, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
            activation()
        )

        self.model_3d = nn.Sequential(
            nn.Conv3d(self.hist_size, d, kernels[0], stride, bias=False),
            activation(),
            nn.ConvTranspose3d(d, 1, kernels[1], stride=2, bias=False),
            nn.ReflectionPad3d((1, 0, 1, 0, 0, 0)),
            activation()
        )

        self.hist = [None] * self.hist_size
        self.last_input = None
        self.iter = 0
        self.picture_every = 100

    def forward(self, x: Tensor) -> Tensor:
        if self.hist[0] is not None and self.hist[0].size() != x.size():
            print("reset history")
            self.hist = [None] * self.hist_size

        if self.hist[0] is None:
            for i in range(self.hist_size):
                self.hist[i] = x
        else:
            for i in range(self.hist_size-1):
                self.hist[self.hist_size - i - 1] = self.hist[self.hist_size - i - 2]
            self.hist[0] = self.last_input

        combined_history = torch.cat(self.hist, -3)
        combined_history, bd = flatten_batch(combined_history, 3)
        y = self.model(combined_history)
        y = unflatten_batch(y, bd)

        # combined_history = torch.stack(self.hist, 2)
        # combined_history, bd = flatten_batch(combined_history, 4)
        # y = self.model(combined_history)
        # y = unflatten_batch(y, bd)
        # y = torch.squeeze(y)

        self.iter += 1
        if self.iter >= self.picture_every:
            try:
                print("Creating pictures New CNN")
                fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                ax1.imshow(np.clip(x.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
                ax1.set_title("Input")
                ax2.imshow(np.clip(y.cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
                ax2.set_title("CNN_out 1")
                ax3.imshow(np.clip(np.mean(np.mean(y.cpu().detach().numpy().astype('float64'), 0), 0).transpose((1,2,0)), 0, 1), interpolation='nearest')
                ax3.set_title("CNN_out mean")
                plt.savefig('pictures/NewCNN_out.png')
                plt.close(fig)

                fig, axs = plt.subplots(math.ceil(self.hist_size/3), 3, squeeze=False)
                for i in range(self.hist_size):
                    axs[math.floor(i/3)][i%3].imshow(np.clip(self.hist[i].cpu().detach().numpy().astype('float64')[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
                    axs[math.floor(i/3)][i%3].set_title(f"x_{i+1}")
                plt.savefig('pictures/history.png')
                plt.close(fig)
                self.iter = 0
            except Exception as e:
                print("found error while creating pictures:")
                print(e)
        self.last_input = x
        

        return y