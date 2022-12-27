# Game class that keeps track of 
* last move, next move, game score, player

# MCST 
how are prob, nn_prob, V, nn_V created, updated?

* explore: descend children (update U)
  * early stop if child has max_U
* backprop
  * update N, V, 
  * update U  if sibling U is not +-inf

* next: find next based on U/temperature


# ConnectN
* ConnectN(size=5, N=4)
* Helper class that keeps track of the game state, check win
* pie rule means you switch sides, but i still not understand how these values and self.player/2.0  -self.player are managed
* make_move at i,j
* switches player between moves, keeps track of n_moves, score, available_moves

# MCST
* mytree = MCTS.Node(copy(ConnectN))
* tree.explore(policy)
* mytreenext, (v, nn_v, p, nn_p) = tree.next(temperature)



```
policies will return a move given a ConnectN game state
note that MCST policy will forget about the previous tree and construct a node based on current level

def Policy_Player_MCTS(game):
    mytree = MCTS.Node(copy(game))
    for _ in range(1000):
        mytree.explore(policy)
       
    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)
    
    return mytreenext.game.last_move

def Random_Player(game):
    return random.choice(game.available_moves())   
```

# How is training done then?

















# elden ring
* miseriecord/
* squareoff/glintblade phalanx
* determination
* spec dex/strength/vigor
* red/blue dagger



