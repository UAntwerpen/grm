import cmath

import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from math import sin, cos, tan, pi, e
from scipy.stats import binom, norm
from scipy.optimize import fsolve
from scipy.integrate import quad


def van_ruwe_gegevens_naar_frequenties(ruwe_gegevens):
    waarden = sorted(list(set(ruwe_gegevens)))
    frequenties = []
    for waarde in waarden:
        frequenties.append(ruwe_gegevens.count(waarde))
    return waarden, frequenties


def van_frequenties_naar_ruwe_gegevens(waarden, frequenties):
    return list(itertools.chain.from_iterable([[value] * freq for value, freq in zip(waarden, frequenties)]))


def to_array(waarden: list) -> np.array:
    return np.array(waarden)


def gemiddelde_van_frequenties(waarden: list, frequenties: list):
    return np.average(waarden, weights=frequenties)


def gemiddelde_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return gemiddelde_van_frequenties(waarden, frequenties)


def mediaan_van_frequenties(waarden: list, frequenties: list):
    order = np.argsort(waarden)
    cdf = np.cumsum(frequenties[order])
    return waarden[order][np.searchsorted(cdf, cdf[-1] // 2)]


def mediaan_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return mediaan_van_frequenties(waarden, frequenties)


def modus_van_frequenties(waarden: list, frequenties: list):  # in the strictest sense, assuming unique mode
    return waarden[np.argmax(frequenties)]


def modus_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return modus_van_frequenties(waarden, frequenties)


def var_(waarden: list, frequenties: list, n: int):
    waarden, frequenties = to_array(waarden), to_array(frequenties)
    avg = gemiddelde_van_frequenties(waarden, frequenties)
    dev = frequenties * (waarden - avg) ** 2
    return dev.sum() / n


def steekproef_variantie_van_frequenties(waarden: list, frequenties: list):
    waarden, frequenties = to_array(waarden), to_array(frequenties)
    return var_(waarden, frequenties, frequenties.sum() - 1)


def steekproef_variantie_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return steekproef_variantie_van_frequenties(waarden, frequenties)


def populatie_variantie_van_frequenties(waarden: list, frequenties: list):
    waarden, frequenties = to_array(waarden), to_array(frequenties)
    return var_(waarden, frequenties, frequenties.sum())


def populatie_variantie_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return populatie_variantie_van_frequenties(waarden, frequenties)


def steekproef_std_van_frequenties(waarden: list, frequenties: list):
    return np.sqrt(steekproef_variantie_van_frequenties(waarden, frequenties))


def steekproef_std_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return steekproef_std_van_frequenties(waarden, frequenties)


def populatie_std_van_frequenties(waarden: list, frequenties: list):
    return np.sqrt(populatie_variantie_van_frequenties(waarden, frequenties))


def populatie_std_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    return populatie_std_van_frequenties(waarden, frequenties)


def staafdiagram_van_frequenties(waarden: list, frequenties: list):
    plt.bar(waarden, frequenties)
    plt.show()


def staafdiagram_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    plt.bar(waarden, frequenties)
    plt.show()


def boxplot_van_frequenties(waarden: list, frequenties: list):
    ruwe_gegevens = van_frequenties_naar_ruwe_gegevens(waarden, frequenties)
    bp_dict = plt.boxplot(ruwe_gegevens,
                          vert=False
                          )
    # teken mediaan
    for line in bp_dict['medians']:
        # get position data for median line
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        plt.text(x, y, '%.1f' % x,
                 horizontalalignment='center')  # draw above, centered
    # teken Q1 en Q3
    # onthoud y-waarde
    y_val = None
    for line in bp_dict['boxes']:
        x, y = line.get_xydata()[0]  # bottom of left line
        y_val = y
        plt.text(x, y, '%.1f' % x,
                 horizontalalignment='center',  # centered
                 verticalalignment='top')  # below
        x, y = line.get_xydata()[3]  # bottom of right line
        plt.text(x, y, '%.1f' % x,
                 horizontalalignment='center',  # centered
                 verticalalignment='top')  # below

    # teken min
    x, y = min(ruwe_gegevens), y_val
    plt.text(x, y, '%.1f' % x,
             horizontalalignment='center'  # centered
             )  # below

    # teken max
    x, y = max(ruwe_gegevens), y_val
    plt.text(x, y, '%.1f' % x,
             horizontalalignment='center'  # centered
             )  # below
    plt.show()


def boxplot_van_ruwe_gegevens(ruwe_gegevens: list):
    waarden, frequenties = van_ruwe_gegevens_naar_frequenties(ruwe_gegevens)
    boxplot_van_frequenties(waarden, frequenties)


def variatie(n, p):
    resultaat = 1
    for i in range(n, n - p, -1):
        resultaat *= i
    return resultaat


def herhalingsvariatie(n, p):
    return n ** p


def permutatie(n):
    return math.factorial(n)


def combinatie(n, p):
    return math.comb(n, p)


def herhalingscombinatie(n, p):
    return combinatie(n + p - 1, p)


def binpdf(x, n, p):
    return binom.pmf(x, n, p)


def bincdf(x, n, p):
    return binom.cdf(x, n, p)


def normalcdf(van, tot, mu=0, sigma=1):
    normalcdf_x1 = norm.cdf(van, mu, sigma)
    normalcdf_x2 = norm.cdf(tot, mu, sigma)

    return normalcdf_x2 - normalcdf_x1


def invNormLeft(p, mu=0, sigma=1):
    return norm.ppf(p, mu, sigma)


def invNormRight(p, mu=0, sigma=1):
    return norm.ppf(1 - p, mu, sigma)


def invNormCenter(p, mu=0, sigma=1):
    overschot = 1 - p
    helft_overschot = overschot / 2
    return float(invNormLeft(helft_overschot, mu, sigma)), float(invNormRight(helft_overschot, mu, sigma))


def z_test_voor_gemiddelde(x_bar, mu, sigma, n, tail='twee'):
    """
    Voer een Z-test uit met de mogelijkheid voor linkszijdige, rechtszijdige of tweezijdige test.
    :param x_bar: Steekproefgemiddelde
    :param mu: Populatiegemiddelde
    :param sigma: Populatiestandaardafwijking
    :param n: Steekproefgrootte
    :param tail: Soort test ('links', 'rechts', of 'twee')
    :return: P-waarde
    """
    # Berekening van de Z-score
    z_score = (x_bar - mu) / (sigma / np.sqrt(n))

    # Berekening van de p-waarde op basis van de gekozen test
    if tail == 'twee':
        p_value = 1 - norm.cdf(abs(z_score))
    elif tail == 'links':
        p_value = norm.cdf(z_score)
    elif tail == 'rechts':
        p_value = 1 - norm.cdf(z_score)
    else:
        raise ValueError("tail must be 'links', 'rechts', or 'twee'")

    return p_value

def z_test_voor_proportie(p_hat, p, n, tail='twee'):
    """
    Voer een Z-test uit met de mogelijkheid voor linkszijdige, rechtszijdige of tweezijdige test.
    :param p_hat: Steekproefproportie
    :param p: Populatieproportie
    :param n: Steekproefgrootte
    :param tail: Soort test ('links', 'rechts', of 'twee')
    :return: P-waarde
    """
    # Berekening van de Z-score
    z_score = (p_hat - p) / np.sqrt(p * (1 - p) / n)

    # Berekening van de p-waarde op basis van de gekozen test
    if tail == 'twee':
        p_value = 1 - norm.cdf(abs(z_score))
    elif tail == 'links':
        p_value = norm.cdf(z_score)
    elif tail == 'rechts':
        p_value = 1 - norm.cdf(z_score)
    else:
        raise ValueError("tail must be 'links', 'rechts', or 'twee'")

    return p_value


def difference(f1, f2):
    def diff(x):
        return f1(x) - f2(x)

    return diff


def solve_1_functie(f, start=0):
    return fsolve(f, start)


def solve_2_functies(f1, f2, start=0):
    return fsolve(difference(f1, f2), start)


def solve_waarde(f, waarde, start=0):
    def f2(x):
        return waarde

    return solve_2_functies(f, f2, start)


def bepaal_nulpunt(f, start):
    return solve_waarde(f, 0, start)


def los_stelsel_op(A, b):
    A = np.array(A)
    b = np.array(b)

    # Solve the system of equations
    return np.linalg.solve(A, b)


def integraal(f, a, b):
    integral, error = quad(f, a, b)
    return integral


def vkv(a, b, c):
    # Bereken de discriminant
    D = b ** 2 - 4 * a * c

    # Controleer of de discriminant groter dan of gelijk aan nul is
    if D < 0:
        return None  # Geen reële oplossingen
    else:
        # Bereken de twee oplossingen
        x1 = (-b + math.sqrt(D)) / (2 * a)
        x2 = (-b - math.sqrt(D)) / (2 * a)
        return x1, x2


def plot_functie(f, minX=-10, maxX=10, minY=None, maxY=None):
    delta = 0.001
    x = np.arange(minX, maxX, delta)
    plt.plot(x, f(x))
    if minY is not None and maxY is not None:
        plt.ylim(minY, maxY)
    plt.show()


def plot_functies(functies, minX=-10, maxX=10, minY=None, maxY=None):
    delta = 0.001
    x = np.arange(minX, maxX, delta)
    for f in functies:
        plt.plot(x, f(x))
    if minY is not None and maxY is not None:
        plt.ylim(minY, maxY)
    plt.show()


def bgsin(x):
    return math.asin(x)


def bgcos(x):
    return math.acos(x)


def bgtan(x):
    return math.atan(x)


def log(x, base=10):
    return math.log(x, base)


def ln(x):
    return math.log(x)


def exp(x):
    return math.exp(x)


def wortel(x, n=2):
    return x ** (1 / n)


def veelterm(coefficienten, x):
    """

    :param coefficienten: de veelterm 2 x^2 + 3x - 1 wordt gegeven door de list [2,3,-1]
    :param x:
    :return:
    """
    coefficienten = coefficienten[::-1]
    return sum(c * x ** i for i, c in enumerate(coefficienten))


def cotan(x):
    return 1 / tan(x)


def graden_naar_gms(graden):
    d = int(graden)
    md = abs(graden - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return d, m, sd


def graden_naar_radialen(graden):
    return math.radians(graden)


def radialen_naar_graden(radialen):
    return math.degrees(radialen)


def modulus(c):
    return abs(c)


def argument(c):
    return cmath.phase(c)


def vlak(richtingsvector1, richtingsvector2, vertegenwoordiger):
    # Bereken de normaalvector als het kruisproduct van de richtingsvectoren
    normaalvector = np.cross(richtingsvector1, richtingsvector2)

    # Bereken D als het dot-product van de normaalvector en de vertegenwoordiger
    D = np.dot(normaalvector, vertegenwoordiger)

    # De vergelijking van het vlak is nu van de vorm Ax + By + Cz = D
    vergelijking_vlak = list(normaalvector) + [D]

    # zet om naar floats ipv np.float64 en np.int64
    return [float(item) for item in vergelijking_vlak]

# # gooien met een dobbelsteen
# ## ruwe gegevens
# ruwe_gegevens = [1, 2, 2, 1, 2, 3, 4, 5, 4, 3, 4, 5, 6, 6, 2, 3, 2, 1, 5, 4, 4, 5, 6]
#
# # staafdiagram_van_ruwe_gegevens(ruwe_gegevens)
# # plt.show()
# # boxplot_van_ruwe_gegevens(ruwe_gegevens)
# # plt.show()
# print(gemiddelde_van_ruwe_gegevens(ruwe_gegevens))
# print(steekproef_variantie_van_ruwe_gegevens(ruwe_gegevens))
# print(populatie_variantie_van_ruwe_gegevens(ruwe_gegevens))
# print(steekproef_std_van_ruwe_gegevens(ruwe_gegevens))
# print(populatie_std_van_ruwe_gegevens(ruwe_gegevens))
#
# ## frequentietabel
# waarden = [1, 2, 3, 4, 5, 6]
# frequenties = [123, 122, 120, 110, 132, 128]
# # staafdiagram_van_frequenties(waarden, frequenties)
# # plt.show()
# # boxplot_van_frequenties(waarden, frequenties)
# # plt.show()
# print(gemiddelde_van_frequenties(waarden, frequenties))
# print(steekproef_variantie_van_frequenties(waarden, frequenties))
# print(populatie_variantie_van_frequenties(waarden, frequenties))
# print(steekproef_std_van_frequenties(waarden, frequenties))
# print(populatie_std_van_frequenties(waarden, frequenties))
#
# ## telproblemen
# n = 10
# p = 3
# # var
# print("var", variatie(n, p))
# # hvar
# print("hvar", herhalingsvariatie(n, p))
# # perm
# print("perm", permutatie(n))
# # comb
# print("comb", combinatie(n, p))
# # hvar
# print("hcomb", herhalingscombinatie(n, p))
#
# ## binomiaalverdeling
# n = 10  # aantal trials
# p = 0.5  # kans op succes
# k = 5  # aantal successen
# print(binpdf(k, n, p))
# print(bincdf(k, n, p))
#
# ## normaalverdeling
# mu = 0
# sigma = 1
# print(normalcdf(-1, 1))
# print(invNormLeft(0.95, mu, sigma))
# print(invNormRight(0.95, mu, sigma))
# print(invNormCenter(0.90, mu, sigma))
#
#
# ## solver
# def func(x):
#     return x ** 2 - 4
#
#
# def func2(x):
#     return 0
#
#
# print(solve_2_functies(func, func2, 1))
#
# # integraal numeriek oplossen
# print(integraal(func, 0, 1))
#
#
# def func3(x):
#     return x ** 2 - 5 * x + 4
#
#
# def func4(x):
#     return 15
#
#
# def func5(x):
#     return integraal(func3, 0, 1) - integraal(func3, 1, 4) + integraal(func3, 4, x)
#
#
# def func6(x):
#     return integraal(func3, 0, 1) - integraal(func3, 1, 4) + integraal(func3, 4, x) - 15
#
#
# print(solve_2_functies(func5, func4, 5))
# print(solve_1_functie(func6, 5))
# print(solve_waarde(func5, 15, 5))
# print(los_stelsel_op([[3, 2], [1, -4]], [18, -6]))
#
#
# def f(x):
#     return x ** 2
#
#
# # plot_functie(f, minY=-10, maxY=10)
# # plot_functies([f,func3], minY=-10, maxY=10)
# print(bgsin(1))
# print(log(10))
# print(log(10, 10))
# print(ln(e))  # ln
# print(veelterm([1, 0, 2], 2))
# print(wortel(4))
# print(wortel(8, 3))
# print(graden_naar_radialen(90))
# print(radialen_naar_graden(pi))
# # Maak een complex getal met behulp van de complex functie
# c1 = complex(1, 2)
# c2 = complex(3, 4)
# print(c1 + c2)
# print(c1 - c2)
# print(c1 * c2)
# print(c1 / c2)
# print(c1 ** 2)
# # derde wortel
# print(c1 ** (1 / 3))
# # of gebruik de wortel functie
# print(wortel(c1, 3))
#
# # Maak een complex getal door een 'j' achter een getal te plaatsen
# c2 = 1 + 2j
# print(c2)  # Output: (1+2j)
#
# c = complex(-1, 0)
# # Bereken de modulus (absolute waarde)
# print("Modulus:", modulus(c))  # afstand 1 tot de oorsprong
#
# # Bereken het argument (hoek in radialen)
# print("Argument:", argument(c))  # pi
#
# print(vlak([1,-3,2],[1,3,-4],[2,0,8]))
