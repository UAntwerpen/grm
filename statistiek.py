from wiskunde import *

# gooien met een dobbelsteen
## ruwe gegevens
ruwe_gegevens = [1, 2, 2, 1, 2, 3, 4, 5, 4, 3, 4, 5, 6, 6, 2, 3, 2, 1, 5, 4, 4, 5, 6]

staafdiagram_van_ruwe_gegevens(ruwe_gegevens)
boxplot_van_ruwe_gegevens(ruwe_gegevens)
print(gemiddelde_van_ruwe_gegevens(ruwe_gegevens))
print(steekproef_variantie_van_ruwe_gegevens(ruwe_gegevens))
print(populatie_variantie_van_ruwe_gegevens(ruwe_gegevens))
print(steekproef_std_van_ruwe_gegevens(ruwe_gegevens))
print(populatie_std_van_ruwe_gegevens(ruwe_gegevens))

## frequentietabel
waarden = [1, 2, 3, 4, 5, 6]
frequenties = [123, 122, 120, 110, 132, 128]

staafdiagram_van_frequenties(waarden, frequenties)
boxplot_van_frequenties(waarden, frequenties)
print(gemiddelde_van_frequenties(waarden, frequenties))
print(steekproef_variantie_van_frequenties(waarden, frequenties))
print(populatie_variantie_van_frequenties(waarden, frequenties))
print(steekproef_std_van_frequenties(waarden, frequenties))
print(populatie_std_van_frequenties(waarden, frequenties))

# toetsen van hypothesen
## z-toets voor gemiddelde: tweeziijdig
print(z_test_voor_gemiddelde(1.5, 1.4, 0.3, 100))
## z-toets voor gemiddelde: linkszijdig
print(z_test_voor_gemiddelde(1.5, 1.4, 0.3, 100, 'links'))
## z-toets voor gemiddelde: rechtszijdig
print(z_test_voor_gemiddelde(1.5, 1.4, 0.3, 100, 'rechts'))

## z-toets voor proportie: tweeziijdig
print(z_test_voor_proportie(0.6, 0.5, 100))
## z-toets voor proportie: linkszijdig
print(z_test_voor_proportie(0.6, 0.5, 100, 'links'))
## z-toets voor proportie: rechtszijdig
print(z_test_voor_proportie(0.6, 0.5, 100, 'rechts'))