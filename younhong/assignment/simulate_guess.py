# simulate_guess.py
# CrushPython - MontyHall

import random
def simulate_guess(nsim):
    """
    Return an arry of list that is  a strategy for guessing which door 
    a prize is behind. This could be a random strategy, one that always 
    guesses 2, whatever. 
    
    You may use randint() as well here

    nsim is the number of simulations to generate guesses for.

    >>> print(simulate_guess(5))
    array([0, 0, 0, 0, 0])
    >>> print(simulate_guess(5))
    [0, 0, 0, 0, 0]
    """

    assert nsim >= 1

    guesses = [ random.randint(0,2) for i in range(nsim) ]
    
    return guesses
