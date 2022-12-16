from util import *
from tqdm import tqdm
from copy import copy
import numpy as np
import G
from math import sqrt
D = G.D
R = G.R

def simulate(state, K=50):
    ''' Note _state is not modified at all
    '''
    wr2p= {0:0, 1:0, .5:0}
    end, winner = is_terminal(state, r=R)

    if end: 
        wr2p[winner] = K
        return wr2p 

    depth = len( state['history'] )
    for i in range(K):
        #state = deepcopy(_state)
        state = reset(state, depth=depth)
        winner = playout(state)
        wr2p[winner] += 1

    state = reset(state, depth=depth)
    return wr2p



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
    state['history'].append(move)
    state['valid_moves'].remove(move)
    state['board'][ij2i(move)] = plyr
    state['key'][move] = plyr
    return state

def add_node(node,move,key):
    newnode = { 'P':node ,'WN':{'W':1, 'N':2}, 'C':{} }

    if bytes(key) in G.CxWN2key:
        C,WN = G.CxWN2key[bytes(key)]
        newnode['C'] = C
        newnode['WN'] = WN
    else:
        G.CxWN2key[bytes(key)] = ( newnode['C'], newnode['WN'] )

    node['C'][move] = newnode
    newnode['P'] = node
    return newnode

def descend(node, valid_moves):
    #select a move from valid moves
    N = node['WN']['N']
    max_ubc, max_move = 0, None # This will get always replaced since ubc>=0
    for move in valid_moves:
        c = node['C'].get(move)
        if c == None: w,n = 1,2
        else: w,n = c['WN']['W'], c['WN']['N']

        ubc = UCB1( x_i=w/n, n_i=n, N=N )

        if ubc >= max_ubc:
            max_ubc = ubc
            max_move = move

    return max_move


def backprop(wr2p, state, nodehist):
    p = len(state['history'])%2
    L = len(state['valid_moves'])
    #if is_terminal(state, r=config.R)[0]: dist_terminal = 0
    #else: dist_terminal = L

    while( len(nodehist) > 0 ):
        #print('/|', node['stat'], node['P']['stat'])
        node = nodehist.pop()
        node['WN']['W'] += wr2p[p]+.5*wr2p[0.5]
        node['WN']['N'] += wr2p[0]+wr2p[1]+wr2p[0.5]
        p = (p-1)%2
       
        if len(nodehist) <= 0: break
        state = rewind(state)
        #dist_terminal += 1

def expandleaf(STATE):
    valid_moves = STATE['valid_moves']
    p = len(STATE['history'])%2
    end, winner = is_terminal(STATE, r=R)

    if len(valid_moves)>0 and not end:
        movefromleaf = random_sample( valid_moves )
        keyleaf = copy(STATE['key'])
        #print_key(keyleaf)
        keyleaf[movefromleaf] = p
        add_node(node, movefromleaf, keyleaf) # make our leaf node not a leaf node


# MAIN LOOP
#TRAVERSER
#descend tree and select leaf node
#simulate - expand - backprop


STATE = {'valid_moves':set(range(D*D)), 'history':[],'board':2+np.zeros( (D,D), dtype=np.long ), 'key':bytearray((D**2)*[2] )  }
root_node = { 'P':None, 'C':{}, 'WN':{'W':1, 'N':2} }

G.CxWN2key[ bytes(STATE['key']) ] = (root_node['C'], root_node['WN'])

# GOAL: Run basic back prop
# Fine a scheme to train, collect data, use the model to advance the Tree while simulating

#TODO: How areyou handling when the node is terminal, you always are adding a move
# Is it ok to just not add if terminal?
#TODO: Need to go around and think through it some more
# DESCEND to LEAF

for i in tqdm(range(10000)):
    level = 0
    node = root_node
    nodehist = [node]
    while( len(node['C']) != 0 ):
        move = descend(node, STATE['valid_moves'])

        STATE = update(STATE, move)

        c = node['C'].get(move)
        if c==None: add_node(node,move,STATE['key']); node=node['C'][move]
        else: node=c

        nodehist.append(node)

    #SIMULATE BACKPROP
    expandleaf(STATE)


    wr2p = simulate(STATE)
    backprop(wr2p, STATE, nodehist)
    reset(STATE, level)
pickle(root_node,'rootnode')

