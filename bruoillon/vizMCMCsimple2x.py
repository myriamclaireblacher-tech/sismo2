import numpy as np
import matplotlib.pyplot as plt

def plot_mcmc_chain(filename):
    # Lecture séquentielle du fichier binaire en tant que flottants 64 bits (double)
    data = np.fromfile(filename, dtype=np.float64)
    
    # Extraction en séparant les paires : [début:fin:pas]
    llk = data[0::2] # Indices pairs : 0, 2, 4...
    X = data[1::2]   # Indices impairs : 1, 3, 5...
    
    steps = np.arange(len(X))

    # Configuration de l'affichage (2 graphiques superposés partageant l'axe X)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Graphe 1 : Trajectoire du modèle X
    ax1.plot(steps, X, color='blue', linewidth=0.5)
    ax1.set_ylabel("Position (X)")
    ax1.grid(True, alpha=0.3)

    # Graphe 2 : Évolution de la log-vraisemblance
    ax2.plot(steps, llk, color='red', linewidth=0.5)
    ax2.set_xlabel("Pas MCMC (Transitions acceptées)")
    ax2.set_ylabel("Log-vraisemblance")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Exécution ciblée sur la première chaîne froide
plot_mcmc_chain("test_chain_cold_0.bin")