# KEY NOTES:
# Since the parent is not unique, we will store the stats on the child



# ALGO ------------------------------

# Search
pi = n**(1/temp) / sum( n**(1/temp) )

# NN: ARCHITECUTRE
* V head
* conv 1x1 BN relu 
* flatten linear256 relu, linear 1 tanh

* P head
conv 1x1 BN relu flatten linear BxB

# Training
uniformly sampled from all recent 20*N_iter matches

loss: v msq + cross entropy + L2 reg
(z-v)**2 + pi*log(P) + c|model_params|**2
# Evaluation
400 g
mcst 1600 simluations to select each move using temperature->0 (deterministic)
must win margin >55% to assign f1

# Self play N_iter matches using 1600 simulations per move
* Temperature 1 early moves
* Temperature -> 0 later moves
# data (board,pi,z)
