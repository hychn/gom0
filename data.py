import torch
import random


def inp2boards(boards):
    # boards: list of np board
    # returns: tensor of Bx3xDxD
    inp = torch.tensor(boards)
    inp = torch.stack( [inp==0, inp==1, inp==2], dim=1).float()
    return inp

def dataloader(DATA, bsize, sample_size):
    IDX = random.choices(range(len(DATA)), k=sample_size)
    i=0
    while( (i+bsize)<=len(IDX) ):
        bIDX = IDX[i:i+bsize]

        z = torch.tensor( [DATA[idx]['z'] for idx in bIDX] )
        boards = [DATA[idx]['board'] for idx in bIDX]
        inp = inp2boards(boards)
        plyr = [DATA[idx]['plyr'] for idx in bIDX]
        pi = torch.tensor( [DATA[idx]['pi'] for idx in bIDX] )
        #convert these to tensors
        yield {'z':z, 'inp':inp, 'plyr':plyr, 'pi':pi}
        i+=bsize



# Training
#P,v = model(board,plyrs)
#loss = (z-v)**2 + pi*log(P) + c|model_params|**2
# HOw to do regularization
