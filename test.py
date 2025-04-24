import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parametry
liczba_danych = 1000  # Liczba punktów danych
proby = 1000  # Liczba prób do symulacji
alpha = 0.05  # Poziom istotności

# Funkcja do obliczania błędu I rodzaju
def oblicz_blad_I_rodzaju(n, proby, alpha):
    odrzucenia_pearson = 0
    odrzucenia_spearman = 0
    
    for _ in range(proby):
        # Generowanie danych z rozkładu normalnego o różnych parametrach
        x = np.random.normal(loc=0, scale=1, size=n)
        y = np.random.normal(loc=0, scale=1, size=n)
        
        # Testowanie korelacji Pearsona
        _, p_pearson = stats.pearsonr(x, y)
        if p_pearson < alpha:
            odrzucenia_pearson += 1
        
        # Testowanie korelacji Spearmana
        _, p_spearman = stats.spearmanr(x, y)
        if p_spearman < alpha:
            odrzucenia_spearman += 1
    
    # Procent odrzuconych hipotez zerowych (błąd I rodzaju)
    bled_I_rodzaju_pearson = odrzucenia_pearson / proby * 100
    bled_I_rodzaju_spearman = odrzucenia_spearman / proby * 100
    
    return bled_I_rodzaju_pearson, bled_I_rodzaju_spearman

# Przeprowadzanie analizy
bled_I_rodzaju_pearson, bled_I_rodzaju_spearman = oblicz_blad_I_rodzaju(liczba_danych, proby, alpha)

# Wyświetlanie wyników
print(f"Błąd I rodzaju (Pearson): {bled_I_rodzaju_pearson:.2f}%")
print(f"Błąd I rodzaju (Spearman): {bled_I_rodzaju_spearman:.2f}%")

# Możemy także przeprowadzić analizę dla różnych rozkładów
# Zmienność błędu I rodzaju w zależności od parametrów rozkładu
parametry = [(0, 1), (0, 2), (1, 1), (1, 2)]  # różne pary średnia i odchylenie standardowe
bledy_pearson = []
bledy_spearman = []

for mu, sigma in parametry:
    bled_pearson, bled_spearman = oblicz_blad_I_rodzaju(liczba_danych, proby, alpha)
    bledy_pearson.append(bled_pearson)
    bledy_spearman.append(bledy_spearman)

# Rysowanie wykresu
fig, ax = plt.subplots()
ax.plot([f'{mu},{sigma}' for mu, sigma in parametry], bledy_pearson, label="Pearson")
ax.plot([f'{mu},{sigma}' for mu, sigma in parametry], bledy_spearman, label="Spearman")
ax.set_xlabel('Parametry rozkładu (mu, sigma)')
ax.set_ylabel('Błąd I rodzaju [%]')
ax.set_title('Błąd I rodzaju w teście istotności dla różnych parametrów rozkładu')
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
