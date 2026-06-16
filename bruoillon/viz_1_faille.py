import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

def plot_all_mcmc_chains():
    # 1. Récupération et tri des fichiers (ordre décroissant)
    bin_files = sorted(glob.glob("test_chain_cold_*.bin"), reverse=True)
    if not bin_files:
        print("Erreur : Aucun fichier 'test_chain_cold_*.bin' trouvé.")
        return

    # Configuration de l'affichage
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(bin_files)))

    param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]
    
    for idx, filename in enumerate(bin_files):
        # Déterminer si c'est la chaîne 0 pour l'épaisseur du trait
        is_chain_zero = "chain_cold_0.bin" in filename
        line_width = 2.0 if is_chain_zero else 0.8
        alpha_val = 1.0 if is_chain_zero else 0.5
        zorder = 10 if is_chain_zero else 1 # Garantit que la chaîne 0 est au-dessus
        
        data = np.fromfile(filename, dtype=np.float64)
        n_steps = len(data) // 13
        if n_steps == 0: continue
        
        data = data[:n_steps * 13].reshape((n_steps, 13))
        steps = np.arange(n_steps)
        
        llk = data[:, 0]
        params = [data[:, 7], data[:, 8], data[:, 9], data[:, 10], data[:, 6]]
        
        # Plot LLK
        axes[0].plot(steps, llk, color=colors[idx], linewidth=line_width, 
                     alpha=alpha_val, zorder=zorder, label=filename)
        
        # Plot paramètres
        for i in range(5):
            axes[i+1].plot(steps, params[i], color=colors[idx], linewidth=line_width, 
                           alpha=alpha_val, zorder=zorder)

    # Mise en forme
    axes[0].set_ylabel("Log-Likelihood", fontweight='bold')
    axes[0].set_title("Superposition des traces MCMC (Ordre décroissant)", fontweight='bold', fontsize=14)
    axes[0].legend(fontsize='x-small', loc='best')

    for i in range(5):
        axes[i+1].set_ylabel(param_names[i], fontweight='bold')
        axes[i+1].grid(True, linestyle=":", alpha=0.6)
    
    axes[-1].set_xlabel("Pas MCMC", fontweight='bold')
    plt.tight_layout()

    # Graphique Fit
    if os.path.exists("final_results.csv"):
        df = pd.read_csv("final_results.csv")
        fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
        ax_fit.plot(df["Time"], df["Target"], 'k--', label="Données (Target)", linewidth=2)
        ax_fit.plot(df["Time"], df["Prediction"], 'r-', label="Modèle Final (Pred)", linewidth=1.5)
        ax_fit.set_title("Comparaison : Cible vs Meilleur Modèle", fontweight='bold')
        ax_fit.set_xlabel("Temps (t_list)")
        ax_fit.grid(True, linestyle=":", alpha=0.6)
        ax_fit.legend()
        plt.tight_layout()

    plt.show()

plot_all_mcmc_chains()