from wiskunde import *

# werken met de basisfuncties
print(bgsin(1))
print(log(10))
print(log(10, 10))
print(ln(e))  # ln
print(veelterm([1, 0, 2], 2))  # voor de veelterm 1 x^2 + 0 x + 2
print(wortel(4))
print(wortel(8, 3))  # de derde wortel van 8
print(graden_naar_radialen(90))
print(radialen_naar_graden(pi))


## vergelijkingen oplossen
def func(x):
    return x ** 2 - 4

bepaal_nulpunt(func,1)

def func2(x):
    return 5

# los op: func = func2 en begin bij 1
print(solve_2_functies(func, func2, 1))

# los op: func = 0 en begin bij 5
print(solve_1_functie(func, 5))

# los op: func = 5 en begin bij 5
print(solve_waarde(func, 5, 5))


def f(x):
    return x ** 2

# teken de functie f met x-waarden van -10 tot 10, de y-waarden zijn passend
plot_functie(f)

# teken de functie f met x-waarden van -100 tot 100, de y-waarden zijn passend
plot_functie(f,minX=-100,maxX=100)

# teken de functie f met y-waarden van -5 tot 5 (en de x-waarden van -10 tot 10)
plot_functie(f, minY=-5, maxY=5)

# teken de functie f met y-waarden van -10 tot 10 (en de x-waarden van -1 tot 20)
plot_functie(f, minX=-10,maxX=20,minY=-10, maxY=10)

# plot meerdere functies op 1 figuur
plot_functies([f, func], minY=-10, maxY=10)

# integraal numeriek oplossen
print(integraal(func, 0, 1)) # de integraal van func van 0 tot 1
