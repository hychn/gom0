from importlib import reload
import torch
from torch import nn

class model(nn.Module):

    def __init__(self, D):
        super(model, self).__init__()
        self.D = D

        k = 2


        self.res1 = BasicBlock(3, k*64, stride=1, ident=False)
        self.res2 = BasicBlock(k*64, k*64, stride=1)
        self.res3 = BasicBlock(k*64, k*64, stride=1)
        self.res4 = BasicBlock(k*64, k*64, stride=1)

        self.phead1 = nn.Sequential(
                nn.Conv2d(k*64, k*64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(k*64),
                nn.ReLU6(),
                nn.Dropout(.5),
                nn.Conv2d(k*64, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU6(),
                nn.Dropout(.5),
                )
        self.phead2 = nn.Sequential(
            nn.Linear(D*D*64, D*D),
            nn.ReLU6(),
            )

        self.vhead1 = nn.Sequential(
                nn.Conv2d(k*64, k, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(k),
                nn.ReLU6(),
                )
        self.vhead2 = nn.Sequential(
                nn.Linear(k*D*D, 256),
                nn.ReLU6(),
                nn.BatchNorm1d(256),
                nn.Dropout(.5),
                nn.Linear(256, 1),
                nn.Tanh(),
                )

        #self.phout = nn.Reu
        
    def forward(self, x, plyrs):
        # plyr will be used to permute
        for i,p in enumerate(plyrs):
            if p==1: x[i,[0,1]] = x[i,[1,0]]

        B = x.shape[0]
        D = self.D

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)


        p = self.phead1(x)
        p = p.reshape(B,-1)
        p = self.phead2(p)+.00001

        v = self.vhead1(x)
        v = v.reshape(B,-1)
        v = self.vhead2(v).reshape(B)
        return p, v
        # outputs [B,D*D], [B]


#TODO: add basic rnet tower
#TODO: cleanup pheads by using a resblock subclass


def conv3x3(chn_in, chn_out, stride=1):
  #return nn.Conv2d(chn_in, chn_out, kernel_size=3, stride=stride, padding=1, bias=True)
  return nn.Conv2d(chn_in, chn_out, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, ident=True):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.ident = ident

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.ident: out = self.relu(out+x)
        else: out = self.relu(out)

        return out

def train(F,DATA, epoch, bsize=32):
    # Train (data) -> f
    # Evaluation (f1,f2) -> pass fail
        # do 1600 search and get pi and make move and do 1600 search with the other one
    import data
    from tqdm import tqdm


    F = F.train().cuda()
    optimizer = torch.optim.Adam(F.parameters(), lr=1e-3,  weight_decay=1e-5)
 
    for e in range(epoch):
        runningloss,i = 0,0
        for b in data.dataloader(DATA, bsize, sample_size=len(DATA)):
            pi,z = b['pi'].cuda(), b['z'].cuda()
            P,v = F(b['inp'].cuda(), b['plyr'])
            #loss = ((z-v)**2).mean() + (pi*torch.log(P)).mean()
            loss = ((z-v)**2).mean() + ( (pi-P)**2 ).mean()
            runningloss += loss.item()
            loss.backward()
            #nn.utils.clip_grad_norm_(F.parameters(), max_norm=50)
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        print('TRAIN epoch:', e, 'avg_loss:', runningloss/i)
		

    F = F.eval().cpu()
    return F

if __name__=='__main__':
    m = model(3)
    x = torch.rand(2,3,3,3)
    p,v = m(x,[0,0])
    p,v = m(x,[1,0])
    print( p.shape, v.shape)
    print(p,v)
    print(p.sum())
