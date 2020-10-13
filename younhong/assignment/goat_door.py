# goat_door.py
# CrushPython - MontyHall

import numpy as np
import random

def goat_door(prizedoors, guesses):
    """ Return the goat door that is opened for each simulation. 
    Each item is 0, 1, or 2, and is different from both prizedoors and guesses.
    
    It simulates the opening of a "goat door" that doesn't contain the prize,
        and is different from the contestants guess
    The prizedoors array or list is the door that the prize is behind 
        in each simulation
    The guesses array or list is the door that the contestant guessed 
        in each simulation
        
    remove(element) removes the first matching element from the list.
    pop() removes the last element from the list.

    >>> print(goat_door(np.array([0, 1, 2]), np.array([1, 1, 1])))
    array([2, 2, 0])              # array([2, 0, 0]) is valid too.
    >>> print(goat_door([0, 1, 2], [1, 1, 1]))
    array([2, 2, 0])              # array([2, 0, 0]) is valid too.
    """
    
    goat_door = []
    
    for i in range(len(prizedoors)):
        notPrizeList = [ j for j in range(3) ]
        
        if prizedoors[i] == guesses[i]:
            notPrizeList.remove(prizedoors[i])
        else:
            notPrizeList.remove(prizedoors[i])
            notPrizeList.remove(guesses[i])
            
        pickGoat = random.choice(notPrizeList)
        goat_door.append(pickGoat)
    return goat_door
