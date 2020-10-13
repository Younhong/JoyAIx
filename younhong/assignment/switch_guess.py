# %load switch_guess.py
# CrushPython - MontyHall

def switch_guess(guesses, goatdoors):
    """ Return the new door after switching. Should be different from 
    both guesses and goatdoors.
    The parameter guesses is an array or list of original guesses for 
        each simulation 
    The parameter goatdoors is an array or list of revealed goat doors for 
        each simulation
    The strategy that always switches a guess after the goat door is opened
    
    remove(element) removes the first matching element from the list.
    pop() removes the last element from the list.
    
    >>> print(switch_guess(np.array([0, 1, 2]), np.array([1, 2, 1])))
    [2, 0, 0]                      # [2, 0, 0] is valid
    >>> print(switch_guess([0, 1, 2], [1, 2, 1]))
    [2, 0, 0]                      # [2, 0, 0] is valid
    """
    switchguess = []

    for i in range(len(guesses)):
        assert guesses[i] != goatdoors[i]

        changeDoor = [ j for j in range(3) ]
        
        changeDoor.remove(goatdoors[i])
        changeDoor.remove(guesses[i])
            
        pickAnotherDoor = random.choice(changeDoor)
        switchguess.append(pickAnotherDoor)
    return switchguess
