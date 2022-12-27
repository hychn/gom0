#notes on MCTS

def processpolicy(policy, ConnectN game):
    returns availablemoves, prob, v


class Node
    def init( game, parent, prob):
        game
        child nodes = {}
        U = 0  numbers for determining which actions to take enxt
        prob = prob
        nn_V  = predicted expectation from neural net
        V = expected V fro MCTS
        N visit count
        outcome = game.score ??? keeps track of guaranteed outcome, stops exploration whe the outcome is certain and there is know perfect play ??? how is this propagated above and below?
        if game score is not None:  
            V= game.score* game.player
            if gamescore=0: U = 0
            else gamescore=0: U =self.V * float('inf') 
    
    def createchild(actions, probs):
        games = [copy(self.game)
        for each action in actions,games: game.move(action)
        child = {action:Node(game,parent=self, probs) }
  
  
    def next(temperature):
        if game.score is not None: raise error (game has ended with score 0:score)
        if not self.child: error no child found and game not ended
        child
        max_U = max(c.U for c in child.values())
        if max_U == float('inf'): 
            # make prob max choose the inf option
            prob = [ 1 if child.U==inf and 0 otherwhilse for c in value.values() ]
        else: 
            # compute prob from N and maxN
            maxN = max(node.N for node in child.values())+1
            prob = torch.tensor([ (node.N/maxN)**(1/temperature) for node in child.values() ])
    
        normalize prob 
        if sum(prob)>0: prob/=sum(prob)
        else: prob = [1/len(child)]*len(child)
    
        #??? what is nn_prob and why is it a collection of child.probs?
        nn_prob = torch.stack([node.prob for node in child.values()])
        nextstate = random.choices( list(child.values), weights=prob )
    
        # V was for the previous player making a move
        # to convert to the current player we add - sign
        return nextstate, (-self.V, -self.nn_v, prob, nn_prob)
        # ??? why do we need the probs if we have already selected next state? why do we also need V? where do we use it?
  
    def detachparent():
        del parent
        self.parent=None
        
    def explore(policy):
        if game.score is not None: return 

        current = self
        while current.child and current.outcome is None:
            maxU = max(c.U for c in current.child)
            actions = [ a for a,c in child.items() if c.U == maxU]
            if len(actions)==0: error zero length actions
            action = random.choice(actions)

            if max_U == -inf:
                currentU = inf; currentV = 1;break
            elif maxU == inf:
                currentU = -inf; currentV = -1;break

            current = child[action]

        
        # node has not been expanded
        if not current.child and current.outcome is None:
            next_actions, probs, v = processpolicy(policy, current.game)
            current.nn_v = -v
            current.create_child(next_actions, probs)
            current.V = -v

        current.N += 1

        # backprop
        while current.parent:
            parent = current.parent
            parent.N +=1
            parent.V += (-current.V - parent.V)/parent.N

            for sibling in parent.child.values():
                if sibling.U!=-inf and sibling.U!=inf:
                    sibling.U = sibling.V + c* sibling.prob * sqrt(parent.N)/(1+sibling.N)
            current = current.parent




# Training
mytree, (v, nn_v, p, nn_p) = mytree.next()
while mytree.outcome is None
    loglist = torch.log(nn_p)*p
    constant = torch.where(p>0, p*torch.log(p),torch.tensor(0.))
    logterm.append(-torch.sum(loglist-constant))

    vterm.append(nn_v*current_player)

loss = torch.sum( (torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
  
#??? how is next and it's return values used in training

# glintblade phlanx 40
# squareoff 40 heavy
# royal knight's resolve 
