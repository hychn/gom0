# TODO:
# Inspect the CW values... 
# Is multiplying by V by - to descend correct?
# MAKE A MAP OF WHAT INFLUENCES WHAT
#is v and z consistant?
#v=-1 when p is loss
# NOTE: how does v and w interplay, esp in subsequent branches
#I"m having a hard time believing that the search is working properly... we really need to dig in and debug

# Why does it take so long to learn the correct move that leads to avoiding a loss even when we have 1600 simulations?
# DUDE ALSO IT SHOULD NOT TAKE T HIS LONG TO CONVERGE THERE IS SOMETHING WRONG GOING ON!!!!!


from util import *
import data
import AI
import torch
from tqdm import tqdm
from copy import copy, deepcopy
import numpy as np
import G
from math import sqrt
D = G.D
R = G.R
#N_GAMES_PER_SELFPLAY= 5000
N_GAMES_PER_SELFPLAY= 500
#N_MCSTSEARCH_PER_MOVE = 1600
N_MCSTSEARCH_PER_MOVE = 30
#DATASET_SIZE = 20*N_GAMES_PER_SELFPLAY* 9 # THIS NEEDS TO BE PER GAMES BUT WE ARE LAZY AND JUST MULT BY 9
N_GAMES_PER_EVAL = 400
DATASET_SIZE = 2*N_GAMES_PER_SELFPLAY* 9 # THIS NEEDS TO BE PER GAMES BUT WE ARE LAZY AND JUST MULT BY 9

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

    if move in node['C']: return node['C'][move]
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
    ubc2move = (D*D)*[0.0]

    for move in valid_moves:
        c = node['C'].get(move)
        
        if c == None: q,n = 0,0
        else: q,n = c['q'], c['n']
        ubc2move[move]  = -q + UCB1( P[move], n, N )

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
        ubc = -q + UCB1( P[move], n, N )

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
        with torch.no_grad(): P,v = F( data.inp2boards([STATE['board']]) , [plyr] )
        P,v = P[0].tolist(),v[0].item()
        node['v'] = v # This expands the node
        node['P'] = P # This expands the node
    else: # terminal with already filled in p,v
        #v = node['q']
        if end: v = .5*node['v'] + .5*-1
        else: v = node['v']

    backprop(v, nodepath)
    reset(STATE, level)


def MCST_selfplay_gendata(node, STATE,F):
    iterDATA = []

    depth = 0

    while True:

        for i in range(N_MCSTSEARCH_PER_MOVE): MCST_search(node,STATE,F, noise_root=True)

        Π = pi(node, temp = 1 if depth<3 else 1/depth)

        Cn = np.array( (D*D)*[0.0] )
        Cw = np.array( (D*D)*[0.0] )
        Cv = np.array( (D*D)*[0.0] )
        for k,v in node['C'].items(): Cw[k]=v['w']
        for k,v in node['C'].items(): Cn[k]=v['n']
        for k,v in node['C'].items(): Cv[k]=v['v']
        ubc = UBC(node, STATE['valid_moves'])
        pi_temp1 = pi(node, temp = 1)
        iterDATA.append( {'board':copy(STATE['board']), 'pi':pi_temp1, 'z':None, 'plyr':depth%2,'Cv':Cv, 'Cn':Cn,'Cw':Cw,'ubc':ubc, 'P':node['P']} )
        
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

    return iterDATA

def MCST_evaluation(F01, node01, node2key01, id01, STATE):
    i = 0
    while True:
        F = F01[i%2]
        G.node2key = node2key01[i%2]
        node = node01[i%2]
        for i in range(N_MCSTSEARCH_PER_MOVE): MCST_search(node,STATE,F, noise_root=True)
        Π = pi(node, temp = 1)
        move = np.argmax(Π)

        STATE = update(STATE,move)

        G.node2key = node2key01[0]
        node01[0] = add_node(node01[0],move,STATE['key'])
        G.node2key = node2key01[1]
        node01[1] = add_node(node01[1],move,STATE['key'])

        end, winner = is_terminal(STATE, R)
        if end: break
        i+=1

    reset(STATE)
    if winner!=.5: return id01[winner]
    else: return winner


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

    F0 = AI.model(D).eval()
    #F0.load_state_dict( torch.load('models/save.pth'))
    STATE = {'plyr':0, 'valid_moves':set(range(D*D)), 'history':[],'board':2+np.zeros( (D,D), dtype=np.long ), 'key':bytearray((D**2)*[2] )  }
    root_node = { 'C':{}, 'n':0, 'q':0, 'w':0, 'P':None }
    G.node2key = {bytes(STATE['key']):root_node }

    for itr in range(100):

        # Self play Data collection (f, node) -> data, node
        for _ in tqdm(range(N_GAMES_PER_SELFPLAY)):
            deltadata = MCST_selfplay_gendata(root_node,STATE,F0)
            DATA += deltadata
        if len(DATA)>DATASET_SIZE: DATA = DATA[-DATASET_SIZE:]

        pickle(DATA, 'data')
        #pickle(root_node, 'rootnode')

        F1 = AI.model(D).eval()
        AI.train(F1, DATA, epoch=10)


        win2plyr = {0:0, .5:0, 1:0}
        root_node0 = root_node
        root_node1 = { 'C':{}, 'n':0, 'q':0, 'w':0, 'P':None }
        node2key0 = G.node2key
        node2key1 = {bytes(STATE['key']):root_node }

        for i in range(N_GAMES_PER_EVAL):
            if i%2==0: winner = MCST_evaluation( [F0,F1], [root_node0, root_node1], [node2key0, node2key1], [0,1], STATE)
            else: winner = MCST_evaluation( [F1,F0], [root_node1, root_node0], [node2key1, node2key0], [1,0], STATE)
            win2plyr[winner]+=1
        
        print('RESULT_EVALUATION', win2plyr)
        if (win2plyr[0]+win2plyr[1]>0) and win2plyr[1]/(win2plyr[0]+win2plyr[1]) > .55:
            print('SUCCEEDED')
            F0 = F1
            root_node = { 'C':{}, 'n':0, 'q':0, 'w':0, 'P':None }
            G.node2key = {bytes(STATE['key']):root_node }
            torch.save(F0.state_dict(), 'models/'+str(itr)+'.pth')
        else:
            pass
            root_node = { 'C':{}, 'n':0, 'q':0, 'w':0, 'P':None }
            G.node2key = {bytes(STATE['key']):root_node }
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





