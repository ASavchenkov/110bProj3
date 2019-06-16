import torch
import torch.nn as nn
from utils import *


def generate_generic_layer():
    return nn.Sequential()
class GenericLayer(nn.Module):
    def __init__(self, first, last):
        super(GenericLayer, self).__init__()
        self.P1 = nn.Sequential(
                nn.Conv2d(first, last, kernel_size=1,padding=0),
                nn.ReLU()
                )
        self.P3 = nn.Sequential(
                nn.Conv2d(first, last, kernel_size=3,padding=1),
                nn.ReLU()
                )
        self.P5 = nn.Sequential(
                nn.Conv2d(first, last, kernel_size=5,padding=2),
                nn.ReLU()
                )
    def forward(self, p):
        p1 = self.P1(p)
        p3 = self.P3(p)
        p5 = self.P5(p)
        return torch.cat((p1,p3,p5),1)


# Join three networks in one module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prep = nn.Sequential(
                GenericLayer(3,50),
                GenericLayer(150,50),
                GenericLayer(150,50),
                )
        self.hiding = nn.Sequential(
                GenericLayer(153,50),
                GenericLayer(150,50),
                nn.Conv2d(150,3,kernel_size=1,padding=0)
                )
        self.reveal = nn.Sequential(
                GenericLayer(3,50),
                GenericLayer(150,50),
                nn.Conv2d(150,3,kernel_size=1,padding=0)
                )

    def forward(self, secret, cover):

        x_1 = self.prep(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2 = self.hiding(mid)
        x_3 = self.reveal(x_2)
        return x_2, x_3
