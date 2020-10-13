# win_percentage.py
# CrushPython - MontyHall

def win_percentage(guesses, prizedoors):
    """ Return the win percentage between 0 and 100. 
    win_percentage is calculated by the percent of times that a simulation of 
        guesses is correct
    The parameters: 
    guesses is an array or list of guesses for each simulation
    prizedoors is an array or list of the location of prize for each simulation

    >>> print(win_percentage(np.array([0, 1, 2]), np.array([0, 0, 0])))
    33.333
    >>> print(win_percentage([0, 1, 2], [0, 0, 0]))
    33.333
    """
    
    wins = []
    winpercentage = 0.0

    for i in range(len(guesses)):
        if guesses[i] == prizedoors[i]:
            wins.append(guesses[i])
    winpercentage = round(len(wins) / len(guesses) * 100, 3)
    return winpercentage
