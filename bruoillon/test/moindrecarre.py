import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd

# Remplissez avec vos vraies données
A = np.random.randn(6, 450)
B = np.random.randn(6, 300)

"""
# Calcul du pseudo-inverse de A
A_pinv = np.linalg.pinv(A)

# Calcul de la solution
X_pinv = A_pinv @ B


# Paramètre de régularisation (à ajuster selon le bruit de vos données)
alpha = 0.1 

# Matrice d'identité 6x6
I = np.eye(A.shape[0])

# Formule optimisée pour les systèmes sous-déterminés
X_tikhonov = A.T @ np.linalg.inv(A @ A.T + alpha * I) @ B



# Décomposition en valeurs singulières
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Choix du rang de troncature r (doit être <= 6)
r = 4 

# Création de la matrice inverse des valeurs singulières tronquées
S_inv_trunc = np.zeros_like(S)
S_inv_trunc[:r] = 1.0 / S[:r]

# Reconstitution du pseudo-inverse tronqué et calcul de la solution
A_pinv_trunc = Vt.T @ np.diag(S_inv_trunc) @ U.T
X_tsvd = A_pinv_trunc @ B

"""
print("Lecture de la matrice G (A)...")
# np.loadtxt lit les valeurs séparées par des espaces ou retours à la ligne.
# On s'assure de lui redonner la forme (6 lignes, 450 colonnes) attendue par le C++
try:
    A = np.loadtxt("../../../../FortranCodes_Myriam/GreensFunctionsV1/G_matrix.txt")
    if A.shape != (6, 450):
        A = A.reshape((6, 450))
    print(f"Forme de A : {A.shape}")
except FileNotFoundError:
    print("Erreur : Fichier G_matrix.txt introuvable. Remplacez par le bon chemin.")
    # Matrice factice pour tester le code si le fichier n'est pas là
    A = np.random.randn(6, 450) 

print("Lecture des réponses de surface (B)...")
# Pandas est parfait pour lire le CSV avec en-têtes généré par votre C++
try:
    df = pd.read_csv("../../surface_responses.csv")
    
    # Extraction du vecteur temps
    t_list = df['Time'].values
    
    # Extraction des 6 colonnes de données (on ignore la colonne Time)
    # Le dataframe a une taille de (300, 6), on le transpose pour avoir (6, 300)
    B = df.drop(columns=['Time']).values.T
    print(f"Forme de B : {B.shape}")
except FileNotFoundError:
    print("Erreur : Fichier surface_responses.csv introuvable.")
    t_list = np.linspace(0, 5, 300)
    B = np.random.randn(6, 300)

# ==========================================
# 2. RÉSOLUTION DU PROBLÈME (Inversion)
# ==========================================
print("Calcul de la solution X (Inversion de Tikhonov)...")

# Paramètre de régularisation (à ajuster selon le bruit de vos données synthétiques)
alpha = 0.01 

# Formulation duale (inversion d'une matrice 6x6 au lieu de 450x450)
I = np.eye(A.shape[0])
X_solution = A.T @ np.linalg.inv(A @ A.T + alpha * I) @ B

print(f"Forme de la solution X : {X_solution.shape} (Attendu: 450x300)")

plt.figure(figsize=(12, 8))

# On utilise imshow pour afficher la matrice comme une image
# aspect='auto' permet d'étirer l'image pour remplir le graphique
# extent=[x_min, x_max, y_max, y_min] cale les bons axes
im = plt.imshow(X_solution, aspect='auto', cmap='viridis', 
                extent=[t_list[0], t_list[-1], 449, 0])

# Ajout d'une barre de couleur pour lire les amplitudes
cbar = plt.colorbar(im)
cbar.set_label('Amplitude estimée (X)')

plt.title("Carte d'évolution de toutes les sous-failles au cours du temps")
plt.xlabel("Temps (s)")
plt.ylabel("Index de la sous-faille (0 à 449)")
plt.tight_layout()

plt.show()

G=B
# ==========================================
# 2. PLOT 1 : LA MATRICE ENTIÈRE (6 x 450)
# ==========================================
# Comme la matrice est 75 fois plus large que haute, aspect='equal' (par défaut)
# rendrait le plot illisible (une ligne fine). On utilise aspect='auto'.

plt.figure(figsize=(15, 4))
# 'seismic' est bon pour les fonctions de Green car le blanc = 0, 
# le rouge = positif, le bleu = négatif.
im1 = plt.imshow(G, cmap='seismic', aspect='auto') 

plt.colorbar(im1, label='Valeur de G(i,j)')

plt.title("Visualisation complète de la matrice G (6 lignes x 450 colonnes)")
plt.xlabel("Index de la colonne (j : Sous-faille 0 à 449)")
plt.ylabel("Index de la ligne\n(i : Station/Composante 0 à 5)")

# Forcer l'affichage de tous les indices de ligne (0 à 5)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.tight_layout()


# ==========================================
# 3. PLOT 2 : ZOOM SUR LE COIN (6 lignes x 20 colonnes)
# + LECTURE DES VALEURS
# ==========================================


# Extraction du coin (toutes les lignes, 20 premières colonnes)
G_zoom = G[:, :20]

# Création d'une figure plus grande pour accueillir le texte
fig, ax = plt.subplots(figsize=(16, 6))

# Affichage de l'image zoomée
# Ici on peut remettre aspect='equal' ou 'auto', 'auto' remplit mieux l'espace
im2 = ax.imshow(G_zoom, cmap='seismic', aspect='auto')

# --- AJOUT DU TEXTE (Pour lire les valeurs) ---
# On parcourt les lignes et les colonnes du zoom
rows, cols = G_zoom.shape
for i in range(rows):
    for j in range(cols):
    	# Récupération de la valeur
        val = G_zoom[i, j]
        # Choix de la couleur du texte (noir si fond clair, blanc si fond foncé)
        # Basé sur la valeur absolue pour simplifier
        color_text = "white" if abs(val) > (np.max(np.abs(G_zoom))*0.7) else "black"
        
        # Ajout du texte au centre de la case (i,j)
        #On formate à 2 décimales pour que ça rentre
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                color=color_text, fontsize=8)

plt.colorbar(im2, label='Valeur')

plt.title("Zoom sur le coin supérieur gauche de G (6x20) avec valeurs numériques")
plt.xlabel("Index de la colonne (j : 0 à 19)")
plt.ylabel("Index de la ligne (i : 0 à 5)")

# Ajustement des axes
ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))

plt.tight_layout()

# Affichage des deux figures
plt.show()