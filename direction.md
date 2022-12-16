#Control the number of simulations based on possible mv

.1.
10.
0..
 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
with 0 to move but it doesn't make winning move 
because it only has visited this node 3 times
and has not seen the winning move
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This is the key point of mcst that we want to take 
1. guide the branching of new node to good ones
2. guide the descent initially to good ones
But the guide should only serve as an initial suggestion, overridden by imperical evidence

Should the approach be... follow the initial guess by model... until after a certain point, then what, do we need to explore other nodes branching?
There should be a good balance...



# AGZ PAPER ANSWERS BOTH OF THESE CONCERNS
how to test 2 policy? with 
First, the depth of the search may be reduced by position evaluation: truncating the search tree at state s and replacing the subtree below s by an approximate value function v(s) ≈ v∗(s) that predicts the outcome from state s
Second, the breadth of the search may be reduced by sampling actions from a policy p(a|s) that is a probability distribution over possible moves a in position s. For example, Monte-Carlo rollouts 8 search to maximum depth without branching at all, by sampling long sequences of actions for both players from a policy p. Averaging over such rollouts can provide an effective position evaluation, achieving super-human performance in backgammon 8 and Scrabble 9, and weak amateur level play in Go.

In each position s, an MCTS search is executed, guided by the neural network f_theta. (How is it guided?) (what is the MCST Search?)

The MCTS search outputs probabilitiesπ of playing each move. These search proba-biities usually select much stronger moves than the raw move probabilities p of the neural network fθ(s).
MCTS may therefore be viewed as a powerful **policy improvement operator**

 Self-play
 with search – using the improved MCTS-based policy to select each move, then using the game
 winner z as a sample of the value – may be viewed as a powerful **policy evaluation operator**


state: s
move: a
z: end win
NN: f(s) = p,v
MCST: α(s) = Π

Number of visits: N(s,a)
Action Value: Q(s,a) = 1/N * sum(V(sleaf from s))
