import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd # Ajout nécessaire pour lire le CSV

def plot_mcmc_chain(filename):
    if not os.path.exists(filename):
        print(f"Erreur : Le fichier {filename} n'existe pas.")
        return

    # 1. Lecture de tous les doubles
    data = np.fromfile(filename, dtype=np.float64)
    n_steps = len(data) // 13
    if n_steps == 0: return

    data = data[:n_steps * 13].reshape((n_steps, 13))
    
    # 2. Extraction des données
    llk         = data[:, 0]
    V0_         = data[:, 6]
    k_a_sigma   = data[:, 7]
    b_a         = data[:, 8]
    D_c_inv     = data[:, 9]
    Dtau_asigma = data[:, 10]
    steps = np.arange(n_steps)

    # 3. Figures MCMC
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    pt_size = 5.0 
    pt_alpha = 0.3

    axes[0].plot(steps, llk, color='red', marker='.', linestyle='none', markersize=pt_size, alpha=pt_alpha)
    axes[0].set_ylabel("Log-Likelihood", fontweight='bold')
    axes[0].set_title(f"Trace MCMC - Fichier : {filename}", fontweight='bold', fontsize=14)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    params = [k_a_sigma, b_a, D_c_inv, Dtau_asigma, V0_]
    param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]
    colors = ['blue', 'green', 'purple', 'orange', 'teal']

    for i in range(5):
        ax = axes[i+1]
        ax.plot(steps, params[i], color=colors[i], marker='.', linestyle='none', markersize=pt_size, alpha=pt_alpha)
        ax.set_ylabel(param_names[i], fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[-1].set_xlabel("Pas MCMC", fontweight='bold')
    plt.tight_layout()

    # ==========================================================
    # NOUVEAU GRAPHIQUE : Comparaison Data vs Prédiction
    # ==========================================================
    if os.path.exists("final_results.csv"):
        df = pd.read_csv("final_results.csv")
        fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
        
        # Superposition des deux colonnes
        ax_fit.plot(df["Time"], df["Target"], 'k--', label="Données (Target)", linewidth=2)
        ax_fit.plot(df["Time"], df["Prediction"], 'r-', label="Modèle Final (Pred)", linewidth=1.5)
        
        ax_fit.set_title("Comparaison : Cible vs Meilleur Modèle", fontweight='bold')
        ax_fit.set_xlabel("Temps (t_list)")
        ax_fit.set_ylabel("Valeur")
        ax_fit.grid(True, linestyle=":", alpha=0.6)
        ax_fit.legend()
        plt.tight_layout()
    else:
        print("\nAttention : 'final_results.csv' non trouvé, impossible de tracer la comparaison.")

    plt.show()

# Exécution
bin_files = sorted(glob.glob("test_chain_cold_0.bin"))
if bin_files:
    plot_mcmc_chain(bin_files[0])