import torch.nn as nn
class CNN_pre(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*8*8, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

