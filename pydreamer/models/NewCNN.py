from decimal import DivisionImpossible
from typing import Optional, Union
from numpy import size
import torch
import torch.nn as nn
import torch.distributions as D
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .functions import *
from .common import *

class NewCNN(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        padding = 1
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            activation(),
            nn.Conv2d(d, d*2, kernels[1], stride),
            activation(),
            nn.Conv2d(d*2, d*4, kernels[2], stride),
            activation(),
            nn.Conv2d(d*4, d*8, kernels[3], stride),
            nn.Flatten(),
            nn.Linear(1536, 4096),
            nn.Unflatten(-1, (64, 64))
        )

        self.x_0 = None
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None

    def forward(self, x: Tensor) -> Tensor:
        if self.x_0 is None:
            self.x_0 = x
            self.x_1 = x
            self.x_2 = x
            self.x_3 = x
        else:
            self.x_3 = self.x_3
            self.x_2 = self.x_2
            self.x_1 = self.x_1
            self.x_0 = x

        print(f"Size input cnn: {x.size()}")
        combined_history = torch.stack((self.x_1, self.x_2, self.x_3), 2)
        print(f"size after concatonation: {combined_history.size()}")
        combined_history, bd = flatten_batch(combined_history, 3)
        y = self.model(combined_history)
        print(f"size output y: {y.size()}")
        y = unflatten_batch(y, bd)
        print(f"size output y: {y.size()}")
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(np.clip(combined_history.detach().numpy()[0].transpose((1,2,0)), 0, 1), interpolation='nearest')
        ax1.set_title("History input")
        ax2.imshow(np.clip(y.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
        ax2.set_title("CNN_out")
        plt.savefig('pictures/NewCNN_out.png')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        print(f"size: {self.x_1.size()}")
        ax1.imshow(np.clip(self.x_1.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
        ax1.set_title("x_1")
        ax2.imshow(np.clip(self.x_2.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
        ax2.set_title("x_2")
        ax3.imshow(np.clip(self.x_3.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
        ax3.set_title("x_3")
        plt.savefig('pictures/history.png')

        return y