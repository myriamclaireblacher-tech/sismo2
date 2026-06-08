import numpy as np

# Exemple de lecture de la chaîne 0 en Python
NSubFaults = 100  # Ton nombre de sous-failles
# 1 double (llk) + 4 double par sous-faille
num_floats_per_row = 1 + 4 * NSubFaults 

data = np.fromfile("chain_cold_0.bin", dtype=np.float64)
# Redimensionne le tableau pour avoir le nombre d'itérations en lignes
data = data.reshape(-1, num_floats_per_row)

llk = data[:, 0]         # Première colonne = toutes les llk
models = data[:, 1:]     # Le reste = tes paramètres physiques