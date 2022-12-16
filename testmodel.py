from importlib import reload
import torch
from torch import nn

class model(nn.Module):

    def __init__(self, D):
        super(model, self).__init__()

        k = 2
        self.vh1 = nn.Sequential(
                nn.Conv2d(k*64,k*64, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(),
                nn.BatchNorm2d(k*64),
                nn.Dropout(.5),
                nn.Conv2d(k*64,k*64, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(),
                nn.BatchNorm2d(k*64),
                nn.Conv2d(k*64,k,kernel_size=1, stride=1),
                nn.ReLU6(),
                nn.BatchNorm2d(k),
                )

        self.vout = nn.Sequential(
                nn.Linear(k*D*D, k*64),
                nn.ReLU6(),
                nn.BatchNorm1d(k*64),
                nn.Dropout(.5),
                nn.Linear(k*64, 1),
                #nn.ReLU6(),
                #nn.BatchNorm1d(1),
                nn.Tanh(),
                #nn.Softmax(1)
                )

        self.mask = nn.Sequential(
                nn.Conv2d(3,1, kernel_size=1, stride=1),
                nn.ReLU6(),
                nn.BatchNorm2d(1),
                )
        self.maskout = nn.ReLU6()

        self.ph1 = nn.Sequential(
                nn.Conv2d(3,k*64, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(),
                nn.BatchNorm2d(k*64),
                nn.Dropout(.5),
                )
        self.ph2 = nn.Sequential(
                nn.Conv2d(k*64,k*64, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(),
                nn.BatchNorm2d(k*64),
                )
        self.ph3 = nn.Sequential(
                nn.Conv2d(k*64,k*64,kernel_size=3, stride=1, padding=1),
                nn.ReLU6(),
                nn.BatchNorm2d(k*64),
                )
        self.ph4 = nn.Sequential(
                nn.Conv2d(k*64+3, 1, kernel_size=1, stride=1),
                nn.ReLU6(),
                nn.BatchNorm2d(1),
                nn.ReLU6()
                )
        #self.phout = nn.Reu
        self.D = D
        
    def forward(self, x, plyrs):
        for i,p in enumerate(plyrs):
            if p==1: x[i,[0,1]] = x[i,[1,0]]


        B = x.shape[0]
        D = self.D

        p = self.ph1(x)
        p = self.ph2(p)+p
        vx = self.ph3(p)+p
        p = self.ph4( torch.cat([vx,x], dim=1) )

        v = self.vh1(vx)
        v = v.reshape(B,-1) #(B,D*D)
        v = self.vout(v) #(B,1)
        v = v.reshape(B)


        #mask = (x[:,2:3]==0)
        #p = p*mask

        # APPROACH 1
        #d = (D*D)*torch.ones(B,1,D,D).to(dvc)
        #p = p/d
        #p = p.reshape(B,-1)

        # APPROACH 2
        p = p.reshape(B,-1)
        d = p.sum(-1).unsqueeze(-1).expand(B,D*D)+.000001
        #d = p.sum(-1).unsqueeze(-1).expand(B,D*D)+.001
        p = p/d

        return p, v
        # outputs [B,D*D], [B]

