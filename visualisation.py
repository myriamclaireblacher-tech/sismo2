import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données du fichier CSV
try:
    # On lit le fichier généré par ton code C++
    data = pd.read_csv('build/test.csv')
except FileNotFoundError:
    print("Erreur : Le fichier 'test.csv' n'a pas été trouvé dans le dossier build.")
    print("Assure-toi d'avoir bien exécuté sismo2.exe d'abord.")
    exit()

# Vérification du nom des colonnes dans la console
print("Colonnes lues :", list(data.columns))

# 2. Extraction des variables
# Selon ton ODE.cpp, les colonnes sont : Time, V, Theta
t = data['Time']
V = data['V']

# Si tu veux convertir V en cm/hour pour coller EXACTEMENT au graphique de l'article :
# Actuellement V est en m/heure (car Vinf et V0 ont été divisés par 365*24)
# Pour passer de m/heure à cm/hour, on multiplie par 100
V_cm_hour = V 

# 3. Création du graphique
plt.figure(figsize=(8, 6))

# Tracé de la courbe
plt.plot(t, V_cm_hour, label='Modèle Rate-and-State', color='black', linewidth=2)

# Configuration des axes pour ressembler à l'article
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Slip velocity (cm/hour)', fontsize=12)
plt.title('Évolution de la vitesse de glissement au cours du temps', fontsize=14)
plt.show()
