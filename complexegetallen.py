from wiskunde import *

print(graden_naar_radialen(90))
print(radialen_naar_graden(pi))

# Maak een complex getal door een 'j' achter een getal te plaatsen
c1 = 1 + 2j
c2 = 3 + 4j
print(c1 + c2)
print(c1 - c2)
print(c1 * c2)
print(c1 / c2)
print(c1 ** 2)
# derde wortel
print(c1 ** (1 / 3))
# of gebruik de wortel functie
print(wortel(c1, 3))

print(c2)  # Output: (1+2j)
print(c2.real) # het reele gedeelte
print(c2.imag) # het imaginaire gedeelte

c = -1 # of -1+0j
# Bereken de modulus (absolute waarde)
print("Modulus:", modulus(c))  # afstand 1 tot de oorsprong

# Bereken het argument (hoek in radialen)
print("Argument:", argument(c))  # pi

