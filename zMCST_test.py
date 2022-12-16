from util import *
import data
import testmodel
import AI
import torch
from tqdm import tqdm
from copy import copy, deepcopy
import numpy as np
import G
from math import sqrt
D = G.D
R = G.R
#N_GAMES_PER_ITER= 5000
N_GAMES_PER_ITER= 500
#N_MCSTSEARCH_PER_MOVE = 1600
N_MCSTSEARCH_PER_MOVE = 30
#DATASET_SIZE = 20*N_GAMES_PER_ITER * 9 # THIS NEEDS TO BE PER GAMES BUT WE ARE LAZY AND JUST MULT BY 9
DATASET_SIZE = 4*N_GAMES_PER_ITER * 9 # THIS NEEDS TO BE PER GAMES BUT WE ARE LAZY AND JUST MULT BY 9

def roundlist(L): return [ round(l,3) for l in L]

def playout(state):

    def algo(state):
        move = random_sample(state['valid_moves'])
        return move

    end = False
    while(not end):
        #move = learning_algo(state)
        move = algo(state)
        state = update(state, move)
        end, winner = is_terminal(state, r=R)
    return winner

def update(state, move):
    plyr = len(state['history'])%2
    state['board'][ij2i(move)] = plyr
    state['key'][move] = plyr

    state['history'].append(move)
    state['valid_moves'].remove(move)
    return state

def add_node(node,move,key):
    newnode = { 'C':{}, 'n':0, 'q':0, 'w':0, 'P':None }

    if bytes(key) in G.node2key:
        newnode = G.node2key[bytes(key)]
    else:
        G.node2key[ bytes(STATE['key']) ] = newnode

    node['C'][move] = newnode
    return newnode

def UCB1(p,n,N):
    return 1 * p * sqrt(N)/(1+n)


def UBC(node,valid_moves):
    N = node['n']
    P = node['P']
    ubc2move = (D*D)*[0]

    for move in valid_moves:
        c = node['C'].get(move)
        
        if c == None: q,n = 0,0
        else: q,n = c['q'], c['n']
        ubc2move[move]  = q + UCB1( P[move], n, N )

    return roundlist(ubc2move)


def descend(node, valid_moves, noise):

    N = node['n']


    max_ubc, max_move = -11111, list(valid_moves)[0]# This will get always replaced since ubc>=0
    P = node['P']
    if noise:
        P = np.array(P)
        e = .25
        #P = (1-e)*P +  e*np.random.dirichlet((D*D)*[.3])
        P = (1-e)*P +  e*dirichlet()

    for move in valid_moves:
        c = node['C'].get(move)
        
        if c == None: q,n = 0,0
        else: q,n = c['q'], c['n']
        ubc = q + UCB1( P[move], n, N )

        if ubc >= max_ubc:
            max_ubc = ubc
            max_move = move
    
    return max_move


def backprop(v, nodepath):
    #L = len(state['valid_moves'])
    #if is_terminal(state, r=config.R)[0]: dist_terminal = 0
    #else: dist_terminal = L

    for i,(node) in enumerate(nodepath[::-1]):
        #print('/|', node['stat'], node['P']['stat'])
        node['n'] +=1
        node['w'] += v*(-1)**(i%2)
        node['q'] = node['w']/node['n']



def MCST_search(node,STATE,F, noise_root=False):
    level = len( STATE['history'] )
    nodepath = [node]
    i = 0
    while (node['P'] != None):
        move = descend(node, STATE['valid_moves'], (i==0 and noise_root))
        STATE = update(STATE, move)

        child = node['C'].get(move)
        if child==None: add_node(node,move,STATE['key']); node=node['C'][move]
        else: node=child

        nodepath.append(node)
        end, winner = is_terminal(STATE, R)
        if end: break
        i+=1


    if node['P']==None:
        plyr = len(STATE['history'])%2
        with torch.no_grad():
            P,v = F( data.inp2boards([STATE['board']]) , [plyr] )
        P,v = P[0].tolist(),v[0].item()
        node['P'] = P # This expands the node
    else: # terminal with already filled in p,v
        v = node['q']

    backprop(v, nodepath)
    reset(STATE, level)


def MCST_selfplay_gendata(node, STATE,F):
    iterDATA = []

    depth = 0

    while True:

        for i in range(N_MCSTSEARCH_PER_MOVE): MCST_search(node,STATE,F, noise_root=True)

        Π = pi(node, temp = 1 if depth<3 else 1/depth)

        ubc = UBC(node, STATE['valid_moves'])
        Cn = np.array( (D*D)*[0] )
        Cw = np.array( (D*D)*[0] )
        for k,v in node['C'].items(): Cn[k]=v['n']
        for k,v in node['C'].items(): Cw[k]=v['w']
        Cn = Cn.reshape(D,D); Cw = Cw.reshape(D,D)

        iterDATA.append( {'board':copy(STATE['board']), 'pi':Π, 'z':None, 'plyr':depth%2,'Cn':Cn,'Cw':Cw,'ubc':ubc} )
        
        #move = max( range(len(Π)), key= Π.__getitem__) #move = np.argmax(Π)
        move = np.random.choice( range(D*D), p=Π )

        STATE = update(STATE,move)
        node = node['C'][move]

        depth += 1

        end, winner = is_terminal(STATE, R)
        if end: 
            Π = pi(node, temp = 1 if depth<3 else 1/depth)
            iterDATA.append( {'board':copy(STATE['board']), 'pi':Π, 'z':None, 'plyr':depth%2} )
            break

    reset(STATE)


    if winner==.5:
        for i,d in enumerate(iterDATA[::-1]): d['z'] = 0
    else:
        for i,d in enumerate(iterDATA[::-1]): d['z'] = (-1)**(i+1)
    #iterDATA = [ d for d in iterDATA if d['z']==1 ]

    return iterDATA[-2:]

def MCST_evaluation(F0,F1): pass


#def p(node):

def pi(node, temp):
    N = node['n']

    Π = np.array((D*D)*[0.0])
    #Π = (D*D)*[0]
    for move in range(D*D):
        child = node['C'].get(move)
        if child!=None:
            Π[move] = child['n']**(1/temp) /  N**(1/temp)
    if Π.sum()==0: return Π
    return Π/Π.sum()




if __name__=='__main__':

    DATA = []


    for itr in range(100):

        F = AI.model(D).eval()
        #F = testmodel.model(D).eval()
        STATE = {'plyr':0, 'valid_moves':set(range(D*D)), 'history':[],'board':2+np.zeros( (D,D), dtype=np.long ), 'key':bytearray((D**2)*[2] )  }
        root_node = { 'C':{}, 'n':0, 'q':0, 'w':0, 'P':None }
        G.node2key = {}
        G.node2key[ bytes(STATE['key']) ] = root_node

        # Self play Data collection (f, node) -> data, node
        for _ in tqdm(range(N_GAMES_PER_ITER)): 
            break
            deltadata = MCST_selfplay_gendata(root_node,STATE,F)
            DATA += deltadata
            #print( UBC(root_node, range(9)) )
            #print( [ v['n'] for k,v in root_node['C'].items()] )
            #print( '--------------------------------------------------' )
            #print((deltadata[1]['board']))
            #print((deltadata[-1]['board']), len(deltadata))
        if len(DATA)>DATASET_SIZE: DATA = DATA[-DATASET_SIZE:]

        #pickle(DATA, 'data')
        #pickle(root_node, 'rootnode')
        DATA = unpickle('./data')

        #F = AI.model(D).eval()
        AI.train(F, DATA, epoch=30)
        torch.save(F.state_dict(), 'models/'+str(itr)+'.pth')
        # Train (data) -> f
        # Evaluation (f1,f2) -> pass fail
            # do 1600 search and get pi and make move and do 1600 search with the other one
        

    # I AM EXTREMELY UNSATISTIFIED!!!!!!!!!!!!!!!
    # WITH HOW THE FKIN TREE ITERATES THROUGH THE SAME MATCH AND NOT EXPORING LIKE IT FKN G NEEDS TO 


    #* Test model
    # TODO TEST THE IMPLEMENTED STUFF:
    #* Test 
        #* make sure tree is generating reasonably
        #* make sure data is also got reasonably

    # TODO: How to flip the input based on player for a batch sys?
    # TODO: Implement Training
    # TODO: Test training
        #* inspect training results
    #* Test evaluation





