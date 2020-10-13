# simulate_prizedoor.py
# CrushPython - MontyHall
import random                    # to use random.randint(......)

def simulate_prizedoor(nsim):
    """    
    Return a random array or list of 0s, 1s, and 2s. 
    The length of the array or list is the parameter nsim - a positive integer. 
    nsim represents the number of simulation. 
    
    This function generates a random array of 0s, 1s, and 2s, representing 
    hiding a prize between door 0, door 1, and door. 
    
    Use random.randint(0,2) to generate the numbers of 0, 1, and 2 randomly.

    >>> print(simulate_prizedoor(3))
    array([0, 0, 2])
    >>> print(simulate_prizedoor(7))
    [2, 1, 0, 1, 1, 0, 2]
    """
    assert nsim >= 1
    
    prizedoors = [ random.randint(0,2) for i in range(nsim) ]

    return prizedoors
