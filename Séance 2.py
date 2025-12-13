#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

#Question 5
# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("data/resultats-elections-presidentielles-2022-1er-tour.csv", encoding="utf-8") as fichier:
    contenu = pd.read_csv(fichier)
    print(contenu)

#Question 6
nb_lignes = len(contenu)
nb_colonnes = len(contenu.columns)
# Affichage sur le terminal
print("Nombre de lignes :", nb_lignes)
print("Nombre de colonnes :", nb_colonnes)
types_colonnes = []

#Question 7
for col in contenu.columns:
    dtype = contenu[col].dtype
    if pd.api.types.is_integer_dtype(dtype):
        types_colonnes.append('int')
    elif pd.api.types.is_float_dtype(dtype):
        types_colonnes.append('float')
    elif pd.api.types.is_bool_dtype(dtype):
        types_colonnes.append('bool')
    else:
        types_colonnes.append('str')

#Question 8
print(contenu.dtypes)
print(contenu.columns)
print(contenu.head())

print(contenu.head())
print(contenu.columns)

#Question 9
inscrits = contenu["Inscrits"]
print(inscrits)

#Question 10
total_inscrits = inscrits.sum()
print("Nombre total d'inscrits", total_inscrits)

somme_colonnes=[]

for col in contenu.columns:
    if contenu[col].dtype in ["int64", "float64"]:
        somme_colonnes.append(contenu[col].sum())
print(somme_colonnes)

for col in contenu.columns:
    if contenu[col].dtype in ["int64", "float64"]:
        print(col,":", contenu[col].sum())

#Question 11
for i in range(len(contenu)):
    dept = contenu.loc[i, "Libellé du département"]
    inscrits = contenu.loc[i, "Inscrits"]
    votants = contenu.loc[i, "Votants"]
    plt.figure(figsize=(6,4))
    plt.bar(["Inscrits", "Votants"], [inscrits, votants], color=['blue', 'red'])
    plt.title(f"{dept}")
    plt.ylabel("Nombre de personnes")
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig(f"{dept}.png")
    plt.close()

#Question 12   
    # Récupérer les valeurs
    blancs = contenu.loc[i, "Blancs"]
    nuls = contenu.loc[i, "Nuls"]
    exprimes = contenu.loc[i, "Exprimés"]
    abstentions = contenu.loc[i, "Abstentions"]
    
    valeurs = [blancs, nuls, exprimes, abstentions]
    categories = ["Blancs", "Nuls", "Exprimés", "Abstentions"]
    couleurs = ["lightgrey", "red", "green", "blue"]  # palette personnalisée


 # Création du diagramme circulaire
    plt.figure(figsize=(6,6))
    plt.pie(valeurs, labels=categories, autopct='%1.1f%%', startangle=90, colors=couleurs)
    plt.title(f"Répartition des votes - {dept}")
    plt.tight_layout()
    plt.savefig(f"dossier_images/{dept}.png")
    plt.close()   

#Question 13
dossier_images = "histogrammes"
# Extraire la colonne des inscrits
inscrits = contenu["Inscrits"]

# Créer l’histogramme
plt.figure(figsize=(10,6))
plt.hist(inscrits, bins=30, color="skyblue", edgecolor="black")

# Ajouter les titres et labels
plt.title("Distribution des inscrits par département (1er tour 2022)")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Nombre de départements")
plt.tight_layout()
plt.savefig(f"{dossier_images}/histogrammes.png")
plt.close()










    
  

