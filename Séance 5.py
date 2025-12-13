#coding:utf8

import pandas as pd
import numpy as np
import math
import scipy
import scipy.stats
from scipy.stats import shapiro, norm
import matplotlib.pyplot as plt

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv"))

#Théorie de l'estimation (intervalles de confiance)
print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")

print(donnees.columns)

# Moyennes arrondies
moyennes = {col: round(donnees[col].mean()) for col in donnees.columns}
print("=== Moyennes par colonne (arrondies) ===")
for col, val in moyennes.items():
    print(f"{col} -> Moyenne : {val}")

# Somme des moyennes
somme_moyennes = sum(moyennes.values())
print("\nSomme des moyennes :", somme_moyennes)

# Fréquences échantillon
frequences_echantillon = {col: round(moyennes[col] / somme_moyennes, 2) for col in moyennes}
print("\n=== Fréquences de l'échantillon ===")
for col, val in frequences_echantillon.items():
    print(f"{col} -> {val}")

# Population mère : 
population_mere = {
    "Pour": 852,
    "Contre": 911,
    "Sans opinion": 422
}

somme_pop = sum(population_mere.values())
frequences_population = {col: round(population_mere[col] / somme_pop, 2) for col in population_mere}

# Comparaison
print("\nComparaison des fréquences :")
for col in donnees.columns:  
    print(f"{col} -> Échantillon : {frequences_echantillon[col]} | Population : {frequences_population[col]}")

# Taille de l'échantillon
n = len(donnees)

# Fonction pour calculer intervalle de fluctuation
def intervalle_fluctuation(p, n, z=1.96):
    marge = z * np.sqrt(p * (1 - p) / n)
    return round(p - marge, 3), round(p + marge, 3)

print("\n=== Intervalles de fluctuation (95%) ===")
for col in donnees.columns:
    p_ech = frequences_echantillon[col]
    inf, sup = intervalle_fluctuation(p_ech, n)
    print(f"{col} -> Fréquence échantillon : {p_ech:.2f}, Intervalle : [{inf}, {sup}]")


# 1. Prendre la première ligne (échantillon)
echantillon = donnees.iloc[0]   # première ligne
echantillon_list = list(echantillon)  # conversion en liste native

# 2. Somme de la ligne = taille de l'échantillon
n = sum(echantillon_list)
print("Taille de l'échantillon :", n)

# 3. Fréquences de l'échantillon
frequences = {col: echantillon[col] / n for col in donnees.columns}
print("\nFréquences de l'échantillon :", frequences)

# 4. Fonction intervalle de confiance
def intervalle_confiance(p, n, z=1.96):
    marge = z * np.sqrt(p * (1 - p) / n)
    return round(p - marge, 3), round(p + marge, 3)

# 5. Calcul des intervalles pour chaque opinion
print("\n=== Intervalles de confiance (95%) ===")
for col in donnees.columns:
    p = frequences[col]
    inf, sup = intervalle_confiance(p, n)
    print(f"{col} -> Fréquence : {p:.2f}, Intervalle : [{inf}, {sup}]")

# Charger les deux fichiers
data1 = pd.read_csv("./data/Loi-normale-Test-1.csv")
data2 = pd.read_csv("./data/Loi-normale-Test-2.csv")

col1 = data1.iloc[:,0].values
col2 = data2.iloc[:,0].values

# Test de Shapiro-Wilk
stat1, p1 = shapiro(col1)
stat2, p2 = shapiro(col2)

print("=== Test de Shapiro-Wilk ===")
print(f"Fichier 1 -> W = {stat1:.3f}, p-value = {p1:.3f}")
print(f"Fichier 2 -> W = {stat2:.3f}, p-value = {p2:.3f}")

# Interprétation
alpha = 0.05
if p1 > alpha:
    print("Le fichier 1 suit une loi normale (on ne rejette pas H0).")
else:
    print("Le fichier 1 ne suit pas une loi normale (on rejette H0).")

if p2 > alpha:
    print("Le fichier 2 suit une loi normale (on ne rejette pas H0).")
else:
    print("Le fichier 2 ne suit pas une loi normale (on rejette H0).")

# Visualisation 
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Histogramme + courbe normale ajustée pour fichier 1
axs[0].hist(col1, bins=20, density=True, color="skyblue", edgecolor="black")
mu1, sigma1 = np.mean(col1), np.std(col1)
x1 = np.linspace(min(col1), max(col1), 100)
axs[0].plot(x1, norm.pdf(x1, mu1, sigma1), 'r', lw=2)
axs[0].set_title(f"Fichier 1 (p-value={p1:.3f})")

# Histogramme + courbe normale ajustée pour fichier 2
axs[1].hist(col2, bins=20, density=True, color="lightgreen", edgecolor="black")
mu2, sigma2 = np.mean(col2), np.std(col2)
x2 = np.linspace(min(col2), max(col2), 100)
axs[1].plot(x2, norm.pdf(x2, mu2, sigma2), 'r', lw=2)
axs[1].set_title(f"Fichier 2 (p-value={p2:.3f})")

plt.suptitle("Comparaison des distributions avec courbe normale ajustée")
plt.tight_layout()
plt.show()
