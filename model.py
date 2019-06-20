from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class UG_Recog_Net(nn.Module):

    def __init__(self):
        super(UG_Recog_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(1536, 4096)
        self.drop1 = nn.Dropout()
        self.fct = nn.Linear(4096, 4999)

    def forward(self, x):
        x = x.float() * 2 / 255 - 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.pad(x, (0, 1))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
       
        x = x.view(-1, 1536)
        x = self.drop1(F.relu(self.fc1(x)))

        x = self.fct(x)

        return x


def get():
    return UG_Recog_Net()
