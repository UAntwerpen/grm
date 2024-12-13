from wiskunde import *


## telproblemen
n = 10
p = 3
# var
print("var", variatie(n, p))
# hvar
print("hvar", herhalingsvariatie(n, p))
# perm
print("perm", permutatie(n))
# comb
print("comb", combinatie(n, p))
# hvar
print("hcomb", herhalingscombinatie(n, p))

## binomiaalverdeling
n = 10  # aantal trials
p = 0.5  # kans op succes
k = 5  # aantal successen
print(binpdf(k, n, p))
print(bincdf(k, n, p))

## normaalverdeling
mu = 0
sigma = 1
print(normalcdf(-1, 1))
print(invNormLeft(0.95, mu, sigma))
print(invNormRight(0.95, mu, sigma))
print(invNormCenter(0.90, mu, sigma))
