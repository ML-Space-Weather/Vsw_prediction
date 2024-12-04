import numpy as np
import torch
import torch.nn as nn
from .basic_layers import ResidualBlock
from .attention_module import AttentionModule_stage1, AttentionModule_stage2,\
    AttentionModule_stage3, AttentionModule_stage0
import torch.nn.functional as F

from ipdb import set_trace as st

# def my_einsum(input, weights):

#     batch_size, i_dim, x_dim, y_dim = input.shape
#     weights_dim = weights.shape
    
#     assert weights_dim[0] == weights_dim[1] == weights_dim[2] == weights_dim[3] == x_dim
    
#     output = torch.zeros(batch_size, weights_dim[4], x_dim, y_dim, dtype=weights.dtype, device=weights.device)
    
#     for b in range(batch_size):
#         for i in range(i_dim):
#             for o in range(weights_dim[4]):
#                 for x in range(x_dim):
#                     for y in range(y_dim):
#                         for ix in range(weights_dim[2]):
#                             for jy in range(weights_dim[3]):
#                                 output[b, o, x, y] += input[b, i, ix + x, jy + y] * weights[i, ix, jy, x, o]
        
#     return output

class LstmReg(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.6, output_size=1, num_layers=2):
        super().__init__()
        self.gru1 = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru2 = torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.gru3 = torch.nn.GRU(hidden_size * 2, hidden_size, num_layers, bidirectional=True)
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=False)
        self.reg = torch.nn.Linear(hidden_size, output_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(20, output_size),
        )

    def forward(self, x):
        # check https://meetonfriday.com/posts/d9cbeda0/
        self.gru1.flatten_parameters()
        x, _ = self.gru1(x)
        s, b, h = x.shape
        x = x.reshape([s * b, h])
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


class CNN_second_try(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

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
                      nn.Linear(64*3*3, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.LeakyReLU()
        self.MaxPool2d = nn.MaxPool2d(4)

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


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
                      #nn.Linear(256, 64),
                      #nn.ReLU(),
                      #nn.Dropout(dropout),
                      nn.Linear(256, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_batch(nn.Module):

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
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
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
                      nn.Linear(64*6*6, 256),
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
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

class CNN_batch_multi(nn.Module):

    def __init__(self, 
                 dropout, 
                 n_var, 
                 radius, 
                 outputs):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='zeros',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(128, 256),
                #       nn.Linear(64*6*9+128, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, outputs),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # self.conv1.we
        x = X[:, :, :, :-1]
        y0 = X[:, 0, :128, -1].squeeze()
        # x = self.MaxPool_pre(x)
        # st()
        # x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        # x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # # x = self.MaxPool2d(x)
        # # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # # ipdb.set_trace()
        # x = x.view(x.size(0), -1)
        # x = torch.vstack([x.T, y0.T]).T
        # x = y0
        # st()
        out = self.out(y0)
        return out


class lstm_reg(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.6,
                 output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        # ipdb.set_trace()
        self.gru1 = torch.nn.GRU(input_size, hidden_size,
                                num_layers,
                                batch_first=True
        )  # , bidirectional=True
        self.gru2 = torch.nn.GRU(input_size,hidden_size,
                                 num_layers,
                                 bidirectional=True,
                                 batch_first=True) #, bidirectional=True
        
        self.gru3 = torch.nn.GRU(hidden_size*2,hidden_size,num_layers, bidirectional=True) #, bidirectional=True
        self.lstm1 = torch.nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers,
                                  # bidirectional=True,
                                  batch_first=True) #, bidirectional=True
        
        self.lstm2 = torch.nn.LSTM(hidden_size,hidden_size,num_layers, bidirectional=False) #, bidirectional=True
        
        self.reg = torch.nn.Linear(hidden_size, output_size)
        self.reg2 = torch.nn.Linear(2, output_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(256, 64),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(20, output_size),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self,X):

        # check https://meetonfriday.com/posts/d9cbeda0/
        x = X.squeeze()
        x0 = x[:, :, -1].reshape(-1, 1)
        # x = x[:, :, :-1]
        # y0 = X[:, 0, :128, -1].squeeze()

        self.gru1.flatten_parameters() 
        # import ipdb;ipdb.set_trace()
        # st()    
        x, _ = self.gru1(x)
        # x, _ = self.lstm1(x)
        
        s,b,h = x.shape
        x = x.reshape([s*b, h])
        #ipdb.set_trace()
        # x = self.out(x)
        x = self.reg(x)

        # st()
        # x = x.view(s,b,-1)
        x = torch.vstack([x.T, x0.T]).T
        x = self.reg2(x).squeeze()
        # x = x.reshape([s, b])

        x = x.view(s,b,-1)

        return x



class CNN_lstm(nn.Module):

    def __init__(self, 
                 dropout, 
                 n_var, 
                #  radius, 
                #  input_size, 
                 channel_num,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(4)
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding_mode='zeros',
                # groups=channel_num,
                )
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(64)
        self.gru1 = torch.nn.GRU(192, 
                        hidden_size,
                        num_layers,
                        batch_first=True
        )  # , bidirectional=True
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(16, 256),
                #       nn.Linear(64*6*9+128, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.reg2 = torch.nn.Linear(2, outputs)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, X):

        st()
        x0 = X[:, :128, -1]
        X = X[:, :, :-1]
        X_all = torch.zeros([X.shape[0], 
                             X.shape[1],
                             192])
        # st()
        for i in range(X.shape[1]):
            x = X[:, i].unsqueeze(1)
            st()
            x = self.MaxPool2d(self.LeakyReLU(self.bn1(self.conv1(x))))
            x = self.MaxPool2d(self.LeakyReLU(self.bn2(self.conv2(x))))
            # x = self.MaxPool2d(self.LeakyReLU(self.conv3(x)))
            # x = self.MaxPool2d(self.LeakyReLU(self.conv4(x)))
            # st()
            X_all[:, i] = x.reshape(x.shape[0], -1)
            # st()
        
        # st()
        self.gru1.flatten_parameters()   
        # st()
        x, _ = self.gru1(X_all.to(X.device))
        
        # x = x.reshape([s*b, h])
        # st()
        x = self.out(x)

        # st()
        x = torch.vstack([x.permute(2, 0, 1), x0.permute(2, 0, 1)]).permute(1, 2, 0)
        # st()
        s,b,h = x.shape
        x = x.reshape([s*b, h])
        # st()
        x = self.reg2(x).squeeze()

        out = x.view(s,b,-1)
        # print(out.shape)
        return out


class pure_CNN(nn.Module):

    def __init__(self, 
                 dropout, 
                 n_var, 
                #  radius, 
                #  input_size, 
                 channel_num,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(4)
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding_mode='zeros',
                # groups=channel_num,
                )
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(64)
        self.gru1 = torch.nn.GRU(192, 
                        hidden_size,
                        num_layers,
                        batch_first=True
        )  # , bidirectional=True
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(16, 256),
                #       nn.Linear(64*6*9+128, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.reg2 = torch.nn.Linear(2, outputs)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, X):

        x0 = X[:, :, -1, 0].unsqueeze(2)
        X = X[:, :, :-1] + 1e-2
        X_all = torch.zeros([X.shape[0], 
                             X.shape[1],
                             192])
        # st()
        for i in range(X.shape[1]):
            x = X[:, i].unsqueeze(axis=1)
            # st()
            x = self.MaxPool2d(self.LeakyReLU(self.bn1(self.conv1(x))))
            x = self.MaxPool2d(self.LeakyReLU(self.bn2(self.conv2(x))))
            # x = self.MaxPool2d(self.LeakyReLU(self.conv3(x)))
            # x = self.MaxPool2d(self.LeakyReLU(self.conv4(x)))
            # st()
            X_all[:, i] = x.reshape(x.shape[0], -1)
            # st()
        
        # st()
        self.gru1.flatten_parameters()   
        # st()
        x, _ = self.gru1(X_all.cuda())
        
        # x = x.reshape([s*b, h])
        # st()
        x = self.out(x)

        # st()
        x = torch.vstack([x.permute(2, 0, 1), x0.permute(2, 0, 1)]).permute(1, 2, 0)
        # st()
        s,b,h = x.shape
        x = x.reshape([s*b, h])
        # st()
        x = self.reg2(x).squeeze()

        out = x.view(s,b,-1)
        # print(out.shape)
        return out



class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class SpectralConv2d_DDP(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_DDP, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))

        weights1 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        weights2 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        # self.weights1 = nn.Parameter(weights1)
        # self.weights2 = nn.Parameter(weights2)
        
        self.weights1 = nn.Parameter(torch.view_as_real(weights1))
        self.weights2 = nn.Parameter(torch.view_as_real(weights2))
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # weights1, weights2 = map(torch.view_as_complex, (self.weights1, self.weights2))
        
        # weights1 = self.weights1
        # weights2 = self.weights2
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  
            x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))

        weights1 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        weights2 = self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        self.weights1 = nn.Parameter(weights1)
        self.weights2 = nn.Parameter(weights2)
        
        # self.weights1 = nn.Parameter(torch.view_as_real(weights1))
        # self.weights2 = nn.Parameter(torch.view_as_real(weights2))
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # weights1, weights2 = map(torch.view_as_complex, (self.weights1, self.weights2))
        
        weights1 = self.weights1
        weights2 = self.weights2
        # weights1 = torch.view_as_complex(self.weights1)
        # weights2 = torch.view_as_complex(self.weights2)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  
            x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO_lstm(nn.Module):

    def __init__(self, 
                 dropout, 
                 modes, width,
                #  input_size,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes = modes
        # self.modes2 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(53, 130) # input channel is 3: (a(x, y), x, y)
        self.p = nn.Linear(3, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

        self.gru1 = torch.nn.GRU(2376, 
                        hidden_size,
                        num_layers,
                        batch_first=True)
        
        self.fc1 = nn.Linear(self.width, 16)
        self.fc2 = nn.Linear(16, 1)

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(16, 1),
                #       nn.Linear(64*6*9+128, 256),
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(256, 64),
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.reg2 = torch.nn.Linear(2, outputs)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, X):

        # st()
        x0 = X[:, :, -1, 0].unsqueeze(2)
        X = X[:, :, :] + 1e-2
        # X = X[:, :, :-1] + 1e-2
        X_all = torch.zeros([X.shape[0], 
                             X.shape[1],
                             2376])

        # st()
        for i in range(X.shape[1]):
            x = X[:, i].unsqueeze(axis=3)
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
            x = self.p(x)
            x = x.permute(0, 3, 1, 2)

            x = F.gelu(self.mlp0(self.norm(self.conv0(self.norm(x))))+self.w0(x))
            x = F.gelu(self.mlp1(self.norm(self.conv1(self.norm(x))))+self.w1(x))
            # x = F.gelu(self.mlp2(self.norm(self.conv2(self.norm(x))))+self.w2(x))
            # x = self.MaxPool2d(self.LeakyReLU(self.bn2(self.conv2(x))))
            X_all[:, i] = x.reshape(x.shape[0], -1)
        
        self.gru1.flatten_parameters()   
        # st()
        x, _ = self.gru1(X_all.cuda())
        x = self.out(x)
        s,b,h = x.shape

        # st()
        # x = torch.vstack([x.permute(2, 0, 1), x0.permute(2, 0, 1)]).permute(1, 2, 0)
        # x = x.reshape([s*b, h])
        # x = self.reg2(x).squeeze()

        out = x.view(s,b,-1)
        return out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO_lstm(nn.Module):

    def __init__(self, 
                 dropout, 
                 modes, width,
                #  input_size,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes = modes
        # self.modes2 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(53, 130) # input channel is 3: (a(x, y), x, y)
        self.p = nn.Linear(3, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

        self.gru1 = torch.nn.GRU(2376, 
                        hidden_size,
                        num_layers,
                        batch_first=True)
        
        self.fc1 = nn.Linear(self.width, 16)
        self.fc2 = nn.Linear(16, 1)
        self.attention = nn.MultiheadAttention(embed_dim=self.width, num_heads=1)  # Define attention layer

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(16, 1),
                #       nn.Linear(64*6*9+128, 256),
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(256, 64),
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.reg2 = torch.nn.Linear(2, outputs)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, X):

        # st()
        x0 = X[:, :, -1, 0].unsqueeze(2)
        X = X[:, :, :] + 1e-2
        # X = X[:, :, :-1] + 1e-2
        X_all = torch.zeros([X.shape[0], 
                             X.shape[1],
                             2376])

        # st()
        for i in range(X.shape[1]):
            x = X[:, i].unsqueeze(axis=3)
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
            x = self.p(x)
            x = x.permute(0, 3, 1, 2)

            x = F.gelu(self.mlp0(self.norm(self.conv0(self.norm(x))))+self.w0(x))
            x = F.gelu(self.mlp1(self.norm(self.conv1(self.norm(x))))+self.w1(x))
            # x = F.gelu(self.mlp2(self.norm(self.conv2(self.norm(x))))+self.w2(x))
            # x = self.MaxPool2d(self.LeakyReLU(self.bn2(self.conv2(x))))
            X_all[:, i] = x.reshape(x.shape[0], -1)
        
        self.gru1.flatten_parameters()   
        # st()
        x, _ = self.gru1(X_all.cuda())
        x = self.out(x)
        s,b,h = x.shape

        # st()
        # x = torch.vstack([x.permute(2, 0, 1), x0.permute(2, 0, 1)]).permute(1, 2, 0)
        # x = x.reshape([s*b, h])
        # x = self.reg2(x).squeeze()

        out = x.view(s,b,-1)
        return out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class V_FNO(nn.Module):

    def __init__(self, 
                 dropout, 
                 modes, width,
                #  input_size,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes = modes
        # self.modes2 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(53, 130) # input channel is 3: (a(x, y), x, y)
        self.p = nn.Linear(3, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # self.p = nn.Linear(self.input_size+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.MaxPool2d = nn.MaxPool2d(5)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes )
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)
        self.gru = torch.nn.GRU(2, hidden_size,
                        num_layers,
                        batch_first=True)
        self.fc1 = nn.Linear(self.width, 16)
        self.fc2 = nn.Linear(16, 1)

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(2808, 256),
                #       nn.Linear(64*6*9+128, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 128),
                    #   nn.Sigmoid()
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.reg2 = torch.nn.Linear(128*2, outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.ReLU6 = nn.ReLU6()
        self.LeakyReLU = nn.LeakyReLU()
        # self.abs = torch.abs()

    def forward(self, x):

        # st()
        x0 = x[:, :128, -1]
        x = x[:, :, :-1].unsqueeze(axis=3)
        # st()
        # for i in range(X.shape[1]):
        # x = X[:, i].unsqueeze(axis=3)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x = F.gelu(self.mlp0(self.norm(self.conv0(self.norm(x))))+self.w0(x))
        x = F.gelu(self.mlp1(self.norm(self.conv1(self.norm(x))))+self.w1(x))
        # x = F.gelu(self.mlp2(self.norm(self.conv2(self.norm(x))))+self.w2(x))
        x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.LeakyReLU(self.bn2(self.conv2(x))))
        # X_all[:, i] = x.reshape(x.shape[0], -1)
        # st()
        # self.gru1.flatten_parameters()   
        # st()
        # x, _ = self.gru1(X_all.cuda())
        s,b,_, _ = x.shape
        # st()
        x = x.view(s, -1)
        x = self.out(x).squeeze()

        # st()
        # x0 = (x0 - x0.mean()) / x0.std()
        x = torch.vstack([x.unsqueeze(0), 
                            x0.unsqueeze(0)]).permute(1, 2, 0)
        x = x.reshape([s, -1])
        x = self.reg2(x).squeeze()
        # x = (self.sigmoid(x) - 0.5)*2
        # x = self.ReLU(x)
        # x = torch.abs(x)
        # st()
        # idx = torch.where(x < -200)
        # x[idx] = 0
        # st()
        # out = x.view(s,b,-1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class ComplexLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out
    ):
        super().__init__()
        linear = nn.Linear(dim, dim_out, dtype = torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(linear.weight))
        self.bias = nn.Parameter(torch.view_as_real(linear.bias))

    def forward(self, x):
        weight = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias)
        return F.linear(x, weight, bias)



class SpatialAttention(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(output_channels, 1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.fc(out))
        return out
    

class V_FNO_DDP(nn.Module):

    def __init__(self, 
                 dropout, 
                 modes, width,
                 vr_mean, vr_std,
                #  input_size,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        # self.vr_mean = vr_mean
        # self.vr_std = vr_std
        self.modes = modes
        # self.modes2 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(53, 130) # input channel is 3: (a(x, y), x, y)
        self.p = nn.Linear(3, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # self.p = nn.Linear(self.input_size+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.MaxPool2d = nn.MaxPool2d(5)
        self.conv0 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes )
        self.conv1 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes )
        self.conv2 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes )
        self.conv3 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes )
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)
        self.gru = torch.nn.GRU(2, hidden_size,
                        num_layers,
                        batch_first=True)
        self.fc1 = nn.Linear(self.width, 16)
        self.fc2 = nn.Linear(16, 1)
        # self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1)  # Define attention layer
        # Spatial attention layer
        self.spatial_attention = SpatialAttention(input_channels=width, output_channels=width)

        self.dropout = dropout

        self.out2 = nn.Sequential(
                      nn.Linear(70200, 128),
                    #   nn.Linear(2808, 128),
                      nn.ReLU(),
        )

        self.out1 = nn.Sequential(
                      nn.Linear(69810, 2808),
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(2808, 256),
                    #   nn.Linear(2808, 256),
                    #   nn.Linear(64*6*9+128, 256),
                      nn.LeakyReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(2808, 656),
                    #   nn.Sigmoid()
                      )
        self.reg1 = torch.nn.Linear(128, outputs)
        self.reg2 = torch.nn.Linear(128*2, outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.ReLU6 = nn.ReLU6()
        self.LeakyReLU = nn.LeakyReLU()
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        # self.abs = torch.abs()

    def forward(self, x):

        # st()
        x0 = x[:, :128, -1]
        x = x[:, :, :-1].unsqueeze(axis=3)
        # st()
        # for i in range(X.shape[1]):
        # x = X[:, i].unsqueeze(axis=3)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        # st()
        # x = F.gelu(self.w0(x))
        # x = F.gelu(self.w1(x))
        x = F.gelu(self.mlp0(self.norm(self.conv0(self.norm(x))))+self.w0(x))
        x = F.gelu(self.mlp1(self.norm(self.conv1(self.norm(x))))+self.w1(x))
        # x = F.gelu(self.mlp2(self.norm(self.conv2(self.norm(x))))+self.w2(x))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.LeakyReLU(self.bn2(self.conv2(x))))
        # X_all[:, i] = x.reshape(x.shape[0], -1)
        # st()
        # self.gru1.flatten_parameters()   
        # st()
        # x, _ = self.gru1(X_all.cuda())
        # Apply spatial attention
        attention_weights = self.spatial_attention(x)
        x = x * attention_weights.view(-1, 1, 1, 1)
        s,b,_, _ = x.shape
        # st()
        x = x.view(s, -1)
        st()
        x1 = self.out1(x).squeeze()
        # x1 = x1.T.unsqueeze(2)
        # x1, _ = self.attention(x1, x1, x1)  # Apply self-attention
        # x1 = x1.squeeze().T
        # x1 = (x1-0.5)*2
        # mag = self.out2(x).squeeze()

        # # st() 
        # x = torch.vstack([x0.T, x1.T]).T
        # x = torch.vstack([mag.T, x1.T]).T
        # x = torch.vstack([x1.T, mag.T]).T

        # x = (x-x.min())/(x.max()-x.min())
        # x = (x - 0.5)*2

        # st()
        # try:
            # x = x1 * x0
            # x = (1 + x1) * x0
            # x = (x - self.vr_mean)
        x0 = (x0 - self.vr_mean) / self.vr_std

        try:
            if len(x1.shape) == 2:
                x = torch.vstack([x1.unsqueeze(0), 
                                    x0.unsqueeze(0)]).permute(1, 2, 0).contiguous()
            else:
                x = torch.vstack([x1.unsqueeze(0).unsqueeze(0), 
                                    x0.squeeze().unsqueeze(0).unsqueeze(0)]).permute(1, 2, 0).contiguous()
        except:
            print('x0/x1 shape is {}/{}'.format(x0.shape, x1.shape))
            st()
        x = x.reshape([s, -1])
        x = self.reg2(x).squeeze()
        # x = self.ReLU6(x)*700/6+200
        # x = self.sigmoid(x*10)
        # x = (self.sigmoid(x*10)-0.5)*2
        # st()
        # idx = torch.where(x < -200)
        # x[idx] = 0
        # st()
        # out = x.view(s,b,-1)
            
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



class V_FNO_long(nn.Module):

    def __init__(self, 
                 dropout, 
                 modes, width,
                 vr_mean, vr_std,
                 hidden_size, 
                 num_layers,
                 outputs):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.modes = modes
        self.width = width
        self.padding = 8  # Pad the domain if input is non-periodic

        # Spectral convolution layers
        self.conv0 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes)

        # Fully connected layers
        self.fc1 = nn.Linear(self.width, 16)
        self.fc2 = nn.Linear(16, 1)

        # Instance normalization
        self.norm = nn.InstanceNorm2d(self.width)

        # GRU for time series processing
        self.gru = torch.nn.GRU(2, hidden_size, num_layers, batch_first=True)

        # Output layers
        self.out1 = nn.Sequential(
                      nn.Linear(70200, 2808),
                      nn.ReLU(),
                      nn.Linear(2808, outputs)
                      )
        self.reg2 = nn.Sequential(
                      nn.Linear(outputs*2, outputs),
                      nn.ReLU(),
                      nn.Linear(outputs, outputs)
                      )
        # self.reg2 = torch.nn.Linear(outputs*2, outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.vr_mean = vr_mean
        self.vr_std = vr_std

    def forward(self, x):
        x0 = x[:, :, -1]  # Last time step values
        x = x[:, :180, :-1].unsqueeze(axis=3)  # Remove the last timestep for main processing

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = F.gelu(self.norm(self.conv0(self.norm(x))))
        x = F.gelu(self.norm(self.conv1(self.norm(x))))

        s, b, _, _ = x.shape
        x = x.view(s, -1)
        x = self.out1(x).squeeze()

        # Normalize the last timestep
        x0 = (x0 - self.vr_mean) / self.vr_std

        # st()
        try:
            if len(x.shape) == 2:
                x = torch.vstack([x.unsqueeze(0), 
                                  x0.unsqueeze(0)]).permute(1, 2, 0).contiguous()
            else:
                x = torch.vstack([x.unsqueeze(0).unsqueeze(0), 
                                  x0.squeeze().unsqueeze(0).unsqueeze(0)]).permute(1, 2, 0).contiguous()
        except:
            print('x0/x1 shape is {}/{}'.format(x0.shape, x.shape))
            st()

        x = x.reshape([s, -1])
        x = self.reg2(x).squeeze()
        x = x * self.vr_std +self.vr_mean
        # x = (self.sigmoid(x) - 0.5)
            
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


    
class dV_FNO_long(nn.Module):

    def __init__(self, 
                 dropout, 
                 modes, width,
                 vr_mean, vr_std,
                 outputs):
        super().__init__()

        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.modes = modes
        self.width = width

        # Define layers
        self.p = nn.Linear(3, self.width)  # Input is the previous 10 timesteps + 2 locations
        self.conv0 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d_DDP(self.width, self.width, self.modes, self.modes)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)

        # Spatial attention layer
        self.spatial_attention = SpatialAttention(input_channels=width, output_channels=width)

        # Output layers
        self.out1 = nn.Sequential(
            nn.Linear(255840, 70200),
            nn.ReLU(),
            nn.Linear(70200, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, outputs)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x0 = x[:, :128, -1]
        x = x[:, :, :-1].unsqueeze(axis=3)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = F.gelu(self.mlp0(self.norm(self.conv0(self.norm(x)))) + self.w0(x))
        x = F.gelu(self.mlp1(self.norm(self.conv1(self.norm(x)))) + self.w1(x))

        # Apply spatial attention
        attention_weights = self.spatial_attention(x)
        x = x * attention_weights.view(-1, 1, 1, 1)

        # Flatten and apply final layers
        s, b, _, _ = x.shape
        x = x.view(s, -1)
        x1 = self.out1(x).squeeze()

        return x1

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



class CNN_batch_multi_2h(nn.Module):

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
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
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
                      nn.Linear(64*6*6, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 12),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return F.sigmoid(out)


class CNN_batch_small(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(3)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
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
                      nn.Linear(64*1*1, 256),
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
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_FE_jannis(nn.Module):
    "This CNN is just a first test"
    def __init__(self, n_var, w, h, p_dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_var, 4 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=n_var),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_var),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * n_var, 16 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=4*n_var),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.BatchNorm2d(16 * n_var),
            nn.Flatten(),
            nn.Linear(w * h * 16 * n_var // 16, 128),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(128, 1))

    def forward(self, X):
        return self.net(X)


class CNN_batch_group(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=2 * n_var,
                kernel_size=3,
                stride=1,
                groups=n_var,
                padding_mode='zeros',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=2 * n_var,
                out_channels=4 * n_var,
                kernel_size=1,
                groups=2*n_var,
                stride=1,
                padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                groups=32,
                stride=1,
                padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                groups=64,
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
        # x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        #x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        #x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        #x = self.MaxPool2d(self.ReLU(self.conv4(x)))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out
    
class CNN_group(nn.Module):
    "This CNN is just a first test"
    def __init__(self, n_var, w, h, p_dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_var, 4 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=n_var),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_var),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * n_var, 8 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=4*n_var),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_var),
            nn.MaxPool2d(2),
            nn.Conv2d(8 * n_var, 16 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=8*n_var),
            nn.ReLU(),
            nn.BatchNorm2d(16 * n_var),
            nn.Flatten(),
            nn.Linear(w * h * 16 * n_var // 64, 256),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(64, 1))

    def forward(self, X):
        return self.net(X)

    
    
class Simple_MLP(nn.Module):

    def __init__(self, dropout, n_var, out_num):
        super().__init__()

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(n_var, 1024),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(1024, 64),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, out_num),
                      #nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        out = self.out(x)
        return out
    
class Simple_MLP2(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(n_var, 16),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(16, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
    
        out = self.out(x)
        return out

class CNN_origin(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

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
                      nn.Linear(64*1*1, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.LeakyReLU()
        self.MaxPool2d = nn.MaxPool2d(5)

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_first_try(nn.Module):
    "This CNN is just a first test"
    def __init__(self, n_var, w, h, p_dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_var, 2 * n_var, 3, 1, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(2 * n_var, 4 * n_var, 3, 1, padding=1,
                      padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_var),
            nn.Flatten(),
            nn.Dropout(p=p_dropout),
            nn.Linear(w * h * 4 * n_var // 4, 128),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(128, 1))

    def forward(self, X):
        return self.net(X)


class ResidualAttentionModel_andong_256(nn.Module):
    # for input size 256
    def __init__(self, n_var, output=12):
        super(ResidualAttentionModel_andong_256, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_var, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage0(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=5, padding=1)
        )
        self.fc = nn.Linear(9216, output)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        # out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        # out = self.attention_module3_2(out)
        # out = self.attention_module3_3(out)
        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_andong_64(nn.Module):
    # for input size 64
    def __init__(self, n_var, output=12):
        super(ResidualAttentionModel_andong_64, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_var, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage0(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(4096, output)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        # out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        # out = self.attention_module3_2(out)
        # out = self.attention_module3_3(out)
        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    
class AutoEncoder(nn.Module):
    
    def __init__(self, code_size, n_sample, channels, radius):
        super().__init__()
        self.code_size = code_size
        self.n_sample = n_sample
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(13 * 13 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, channels*radius**2)
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.shape[0], -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([self.n_sample, channels, radius, radius])
        return out
    
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
