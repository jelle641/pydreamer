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
        stride = 1
        padding = 1
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride, padding=2),
            activation(),
            nn.Conv2d(d, 1, kernels[1], stride, padding),
            activation(),
        )

        self.x_0 = None
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None
        self.iter = 0
        self.picture_every = 100

    def forward(self, x: Tensor) -> Tensor:
        if self.x_0 is None:
            self.x_0 = x
            self.x_1 = x
            self.x_2 = x
            self.x_3 = x
        else:
            self.x_3 = self.x_2
            self.x_2 = self.x_1
            self.x_1 = self.x_0
            self.x_0 = x

        combined_history = torch.stack((self.x_1, self.x_2, self.x_3), 2)
        combined_history, bd = flatten_batch(combined_history, 3)
        # y = self.model(combined_history)
        # y = unflatten_batch(y, bd)
        
        x_1_out, bd1 = flatten_batch(self.x_1, 3)
        x_2_out, bd2 = flatten_batch(self.x_1, 3)
        x_3_out, bd3 = flatten_batch(self.x_1, 3)
        y_1 = self.model(x_1_out)
        y_2 = self.model(x_2_out)
        y_3 = self.model(x_3_out)
        y_1 = unflatten_batch(y_1, bd1)
        y_2 = unflatten_batch(y_2, bd2)
        y_3 = unflatten_batch(y_3, bd3)

        y = torch.cat((y_1, y_2, y_3), 2)

        self.iter += 1
        if self.iter == self.picture_every:
            print("Creating pictures")
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(np.clip(combined_history.detach().numpy()[0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax1.set_title("History input")
            ax2.imshow(np.clip(y.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax2.set_title("CNN_out")
            plt.savefig('pictures/NewCNN_out.png')

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(np.clip(self.x_1.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax1.set_title("x_1")
            ax2.imshow(np.clip(self.x_2.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax2.set_title("x_2")
            ax3.imshow(np.clip(self.x_3.detach().numpy()[0][0].transpose((1,2,0)), 0, 1), interpolation='nearest')
            ax3.set_title("x_3")
            plt.savefig('pictures/history.png')
            self.iter = 0
        

        return y