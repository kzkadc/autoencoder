import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def compose(x, funcs):
    y = x
    for f in funcs:
        y = f(y)
    return y


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 1000)
        self.fc2 = nn.Linear(1000, 800)
        self.fc3 = nn.Linear(800, 500)
        self.fc4 = nn.Linear(500, 100)
        self.fc5 = nn.Linear(100, 10)
        self.fc6 = nn.Linear(10, 2)

        self.out_bias = Parameter(torch.zeros(28 * 28, dtype=torch.float))

    def forward(self, x):
        h = self.encode(x.flatten(start_dim=1))
        h = self.decode(h)
        return h

    def encode(self, x):
        return compose(x, [
            self.fc1, F.relu,
            self.fc2, F.relu,
            self.fc3, F.relu,
            self.fc4, F.relu,
            self.fc5, F.relu,
            self.fc6, F.relu
        ])

    def decode(self, c):
        # encoderと重みを共有 (tied-weight)
        w6 = self.fc6.weight.T
        w5 = self.fc5.weight.T
        w4 = self.fc4.weight.T
        w3 = self.fc3.weight.T
        w2 = self.fc2.weight.T
        w1 = self.fc1.weight.T

        h = F.relu(F.linear(c, w6, self.fc5.bias))
        h = F.relu(F.linear(h, w5, self.fc4.bias))
        h = F.relu(F.linear(h, w4, self.fc3.bias))
        h = F.relu(F.linear(h, w3, self.fc2.bias))
        h = F.relu(F.linear(h, w2, self.fc1.bias))
        h = torch.sigmoid(F.linear(h, w1, self.out_bias))

        return h


class AE_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=2, bias=False)
        self.bne1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.bne2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.bne3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0)
        self.bne4 = nn.BatchNorm2d(1)

        self.bnd4 = nn.BatchNorm2d(64)
        self.bnd3 = nn.BatchNorm2d(64)
        self.bnd2 = nn.BatchNorm2d(32)

    def forward(self, x):
        h = self.encode(x.reshape(-1, 1, 28, 28))
        h = self.decode(h)
        return h

    def encode(self, x):
        return compose(x, [
            self.conv1, self.bne1, F.relu,
            self.conv2, self.bne2, F.relu,
            self.conv3, self.bne3, F.relu,
            self.conv4, self.bne4, F.relu
        ])

    def decode(self, c):
        return compose(c, [
            lambda x: F.conv_transpose2d(x, self.conv4.weight),
            self.bnd4, F.relu,
            lambda x: F.conv_transpose2d(x, self.conv3.weight, stride=2),
            self.bnd3, F.relu,
            lambda x: F.conv_transpose2d(x, self.conv2.weight),
            self.bnd2, F.relu,
            lambda x: F.conv_transpose2d(x, self.conv1.weight, stride=2, padding=2),
            torch.sigmoid
        ])
