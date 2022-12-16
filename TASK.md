* .1.
* 00.
* ..x

* Why does 1 try to place at x?


# Multi-Step?

* 11x
* 001
* y.0

* 0 is instructed to place at x, and thinks it won
* thinking that it won is fine but it needs to place at y
* can we incorporate multistep rollback?
* So, after placing moves x,y does the next player have a pretty good idea on who wins?
* If so, can we use that info in a multi-step rollback?

# v BUG
* How is v being calculated? probability of the current player winning
* The v value is also being applied all the way up to the root.
* Is there a better way we can propogate the MCST?


* Trying to make it more sample efficient 
* Code restructure
  * Do I clearly understand what is going on in this program, is it readable?
* I would like to focus on how the tree is expanding and making the decisions it is making.
  * Would it be possible to easily take pieces of the algorithm and setup a separate flow to do this?
    * the branching part
  * i remember I had some nasty code where it was similar things being done is different ways, I would like to prevent this

# Questions

* How are the next moves being chosen?
* when do you balance between exploration and more tight plays?

