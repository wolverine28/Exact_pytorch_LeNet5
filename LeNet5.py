import numpy as np
import torch
import torch.nn as nn
import torch_rbf
import torch.nn.functional as F


class Subsample(nn.Module):
    def __init__(self,f_count):
        super(Subsample, self).__init__()
        # self.S = nn.AvgPool2d(2, stride=2, divisor_override=1)
        self.weight = nn.Parameter(torch.Tensor(1,f_count,1,1))
        self.bias = nn.Parameter(torch.Tensor(1,f_count,1,1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, 0, 1)
        nn.init.normal_(self.bias, 0, 1)

    def forward(self, x):
        x = F.avg_pool2d(x,2,2, divisor_override=1)
        x = x*self.weight+self.bias
        return x

class MaskedConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,mask):
        super(MaskedConv2d, self).__init__()
        self.fm = mask
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        # self.C = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=kernel_size)
        # print()

        self.module_list = nn.ModuleList()
        for h in range(self.in_channels):
            self.module_list.append(MaskedConv2dSingle(self.kernel_size,self.fm[h,:]))

        self.bias = nn.Parameter(torch.Tensor(1,self.out_channels,1,1))
        # nn.init.normal_(self.bias, 0, 1)
        self.weight =nn.Parameter(torch.Tensor(0))

    def forward(self, x):
        out = torch.zeros((x.shape[0],self.out_channels,10,10)).cuda()
        for ch in range(self.in_channels):
            out[:,self.fm[ch,:]==1,:,:] = self.module_list[ch](x[:,[ch],:,:])
        x = out+self.bias
        # x = out
        return x

class MaskedConv2dSingle(nn.Module):
    def __init__(self,kernel_size,singlemask):
        super(MaskedConv2dSingle, self).__init__()
        self.singlemask = singlemask
        self.C = nn.Conv2d(in_channels=1,out_channels=int(singlemask.sum()),kernel_size=kernel_size,bias=False)

    def forward(self, x):
        x = self.C(x)
        return x

class LeNet5(nn.Module):
    def __init__(self,num_list):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.S1 = Subsample(6)
        self.fm = torch.FloatTensor([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
                    [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
                    [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
                    [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
                    [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
                    [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]]).cuda()
        # self.fm = torch.FloatTensor([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
        #             [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
        #             [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
        #             [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
        #             [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
        #             [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]]).cuda()

        self.C3_mask = MaskedConv2d(in_channels=6,out_channels=16,kernel_size=5,mask=self.fm)
        # self.C3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)

        self.S4 = Subsample(16)
        self.C5 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        self.F6 = nn.Linear(120,84)
        self.output = torch_rbf.RBF(84, 10, torch_rbf.linear,num_weights=np.array(num_list))


    def Atanh(self,x):
        return nn.Tanh()(x*2/3)*1.7159

    def forward(self, x):
        x = self.C1(x)
        x = self.Atanh(x)
        x = self.S1(x)
        x = self.Atanh(x)
        x = self.C3_mask(x)
        x = self.Atanh(x)
        x = self.S4(x)
        x = self.Atanh(x)
        x = self.C5(x).view((-1,120))
        x = self.Atanh(x)
        x = self.F6(x)
        x = self.Atanh(x)
        x = self.output(x)
        return x

    def Loss(self,y_ture,y_pred,j=1.):
        tag_onehot = nn.functional.one_hot(y_ture)
        SE = tag_onehot*y_pred
        competive = torch.exp(-y_pred).sum(1)+torch.exp(torch.tensor(j))
        loss = (SE.sum(1)+competive).mean()
        return loss

