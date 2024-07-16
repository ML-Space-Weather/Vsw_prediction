import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.vr_mean = vr_mean
        self.vr_std = vr_std
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
                      nn.Linear(70200, 256),
                    #   nn.ReLU(),
                    #   nn.Dropout(dropout),
                    #   nn.Linear(2808, 256),
                    #   nn.Linear(2808, 256),
                #       nn.Linear(64*6*9+128, 256),
                      nn.LeakyReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 128),
                    #   nn.Sigmoid()
                      )
        self.reg1 = torch.nn.Linear(128, outputs)
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
        # st()
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

        if len(x1.shape) == 2:
            x = torch.vstack([x1.unsqueeze(0), 
                                x0.unsqueeze(0)]).permute(1, 2, 0).contiguous()
        else:
            x = torch.vstack([x1.unsqueeze(0).unsqueeze(0), 
                                x0.squeeze().unsqueeze(0).unsqueeze(0)]).permute(1, 2, 0).contiguous()
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
