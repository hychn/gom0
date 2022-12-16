from functools import lru_cache
from math import sqrt
import sys
import G
from numpy import log as ln
D = G.D
R = G.R


import numpy as np

pregen_diri = [ np.random.dirichlet((D*D)*[.3]) for _ in range(300) ]
def dirichlet():
    idx = random.randint(0, len(pregen_diri)-1)
    return pregen_diri[idx]

def maxidx(L):
    return max( range(len(L)), key= L.__getitem__)

def divide(a,b):
    if b != 0: return a/b
    if b == 0: 
        if a>0: return float('inf')
        if a<0: return -float('inf')
        if a==0: return a/b #ERROR

def depth(node):
    i = 0
    while node['P'] != None: i+=1
    return i


def ij2i(i): return i//D,i%D


def is_terminal(state, r, save=True):
    ''' Check if is terminal based on only LAST MOVE and 'hist' length
    NOTE It is critical that the state has been built up properly
    r means how many moves it takes to win
    '''
    key = bytes(state['key'])
    dterminalw = G.terminal_p2key.get(key)
    if dterminalw != None: return dterminalw

    # Empty Board
    if len(state['history'])==0: 
        RETV = False,None
        if save: G.terminal_p2key[key] = RETV
        return RETV
    
    # Check for winner from the last move
    last_move1d = state['history'][-1]
    last_move = ij2i( last_move1d )
    winner = state['board'][last_move]
    lines = get_lines(last_move, r, D)
    for line in lines:
        if check_win(line, last_move, state['board'], r): 
            RETV = True, winner
            if save: G.terminal_p2key[key] = RETV
            return RETV


    # no winnder but full board
    if len(state['history']) == D**2: 
        RETV = True, .5
        if save: G.terminal_p2key[key] = RETV
        return RETV

    RETV = False, None
    if save: G.terminal_p2key[key] = RETV
    return RETV


@lru_cache(maxsize=2*D**2, typed=False)
def get_lines(xy, r, D):
    ''' xy origin, r radius 
    r=1 is just the point xy
    '''
    x,y = xy
    #k = (D-1)//2
    k=r-1
    delta = list(range(-k,k+1))
    
    l1 = [ (x+d, y+d) for d in delta if inbound(x+d,y+d,D) ]
    l2 = [ (x, y+d) for d in delta if inbound(x,y+d,D)]
    l3 = [ (x+d, y) for d in delta if inbound(x+d,y,D)]
    l4 = [ (x-d, y+d) for d in delta if inbound(x-d,y+d,D)]
    return l1,l2,l3,l4
    
def inbound(x,y,D):
    return  0<=x<D and 0<=y<D
    
def check_win(line, last_move, board, r):
    plyr = board[last_move]
    assert plyr == 0 or plyr == 1
    if len(line)<r: return False

    cnt=0
    for idx in line:
        if plyr==board[idx]: cnt+=1
        else: cnt=0

        if cnt>=r: return True

    return False

def rewind(state):
    lmove = state['history'].pop()
    state['key'][lmove] = 2
    state['board'][ ij2i(lmove) ] = 2
    state['valid_moves'].add(lmove)
    return state

def reset(state, depth=0):
    while len(state['history'])>depth:
        lmove = state['history'].pop()
        state['key'][lmove] = 2
        state['board'][ ij2i(lmove) ] = 2
        state['valid_moves'].add(lmove)
    return state



import pickle as pick
global recursion_limit
recursion_limit = sys.getrecursionlimit()# defalut is 1000
def unpickle(file):
    global recursion_limit
    while True:
        try:
            with open(file, 'rb') as fo:
                data = pick.load(fo, encoding='bytes')
                break
        except RecursionError:
            recursion_limit += 100
            sys.setrecursionlimit(recursion_limit)

    return data

def pickle(data,filename):
    global recursion_limit
    while True:
        try:
            f = open(filename,"wb")
            pick.dump(data,f,protocol=2)
            f.close()
            break
        except RecursionError:
            recursion_limit += 100
            sys.setrecursionlimit(recursion_limit)

import random
def random_sample(X):
    X = tuple(X)
    return random.choice( X )




def print_key(k):
    STR = ''.join( [str(x) for x in k] )
    print(STR)


def probs(node):
    ## TODO INCOMPLETE
    default = 0
    probs = []
    for i in range(D*D):
    #for move,c in node['C'].items():
        c = node['C'].get(i)
        if c!=None:
            probs.append( round( c['WN']['W']/c['WN']['N'], 2) )
        else:
            probs.append( default )

    return probs
 
