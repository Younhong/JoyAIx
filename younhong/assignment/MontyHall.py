# MontyFall.py
# CrushPython - MontyHall

nsim = 1000

prizedoor = simulate_prizedoor(nsim)
guess = simulate_guess(nsim)

print("Winning Rate with the original guess:", win_percentage(guess, prizedoor))

#switch everytime for nsim
goatdoor = goat_door(prizedoor, guess)
switchguess = switch_guess(guess, goatdoor)

print("Winning Rate with the switched guess:", win_percentage(switchguess, prizedoor))
