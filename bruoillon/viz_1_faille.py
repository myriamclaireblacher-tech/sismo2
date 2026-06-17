import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import glob
import os
import pandas as pd

# Valeurs cibles
k_asigma    = 0.01 / 0.4
b_a         = 0.17 / 0.4
D_c_inv     = 1.0 / 0.1
dtau_asigma = 2.0 / 0.4
V0__        = 0.08 * 100.0 / (365.0 * 24.0)
true_values = [k_asigma, b_a, D_c_inv, dtau_asigma, V0__]

def plot_all_mcmc_chains():
    bin_files = sorted(glob.glob("test_chain_cold_*.bin"), reverse=True)
    if not bin_files:
        print("Erreur : Aucun fichier trouvé.")
        return

    # 1. CRÉATION DE LA FIGURE PRINCIPALE
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(bin_files)))
    param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]
    
    # Listes pour stocker les min/max initiaux afin de pré-remplir le panneau
    y_mins = [float('inf')] * 6
    y_maxs = [float('-inf')] * 6

    for idx, filename in enumerate(bin_files):
        is_chain_zero = "chain_cold_0.bin" in filename
        line_width = 2.0 if is_chain_zero else 0.8
        alpha_val = 1.0 if is_chain_zero else 0.5
        zorder = 10 if is_chain_zero else 1
        
        data = np.fromfile(filename, dtype=np.float64)
        n_steps = len(data) // 13
        if n_steps == 0: continue
        
        data = data[:n_steps * 13].reshape((n_steps, 13))
        steps = np.arange(n_steps)
        
        llk = data[:, 0]
        params = [data[:, 7], data[:, 8], data[:, 9], data[:, 10], data[:, 6]]
        
        # Sauvegarde des limites pour le panneau de contrôle
        y_mins[0] = min(y_mins[0], np.min(llk))
        y_maxs[0] = max(y_maxs[0], np.max(llk))
        
        # Plot LLK
        axes[0].plot(steps, llk, color=colors[idx], linewidth=line_width, 
                     alpha=alpha_val, zorder=zorder, label=filename)
        
        # Plot paramètres
        for i in range(5):
            y_mins[i+1] = min(y_mins[i+1], np.min(params[i]))
            y_maxs[i+1] = max(y_maxs[i+1], np.max(params[i]))
            axes[i+1].plot(steps, params[i], color=colors[idx], linewidth=line_width, 
                           alpha=alpha_val, zorder=zorder)

    # Mise en forme de la figure principale
    axes[0].set_ylabel("Log-Likelihood", fontweight='bold')
    axes[0].set_title("Superposition des traces MCMC", fontweight='bold', fontsize=14)
    axes[0].legend(fontsize='x-small', loc='best')

    for i in range(5):
        ax = axes[i+1]
        ax.set_ylabel(param_names[i], fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(y=true_values[i], color='black', linestyle='--', linewidth=1.5)
        if i == 0: ax.legend(fontsize='small')
    
    axes[-1].set_xlabel("Pas MCMC", fontweight='bold')
    plt.tight_layout()

    # ==========================================================
    # 2. CRÉATION DU PANNEAU DE CONTROLE DES AXES Y
    # ==========================================================
    # Fenêtre séparée un peu plus grande pour accueillir les 12 champs
    fig_ctrl = plt.figure("Contrôle des Axes Y", figsize=(6, 8))
    fig_ctrl.text(0.5, 0.96, "Ajustement des échelles verticales (Y)", 
                  ha="center", va="top", fontweight="bold", fontsize=12)

    # Titres des colonnes
    fig_ctrl.text(0.45, 0.91, "Y Min", ha="center", fontweight="bold")
    fig_ctrl.text(0.75, 0.91, "Y Max", ha="center", fontweight="bold")

    # Listes pour conserver les références des TextBox en mémoire (sinon Python les détruit)
    text_boxes_min = []
    text_boxes_max = []
    
    # Noms de toutes les lignes du graphique (LLK + 5 paramètres)
    row_names = ["Log-Likelihood", "k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]

    # Génération automatique des 12 boîtes de saisie
    for i in range(6):
        y_pos = 0.82 - (i * 0.13) # Calcul de la position verticale de la ligne
        
        # Label de la ligne
        fig_ctrl.text(0.05, y_pos + 0.03, row_names[i], va="center", fontweight="bold", fontsize=10)
        
        # Axes pour les zones de texte [gauche, bas, largeur, hauteur]
        ax_min = fig_ctrl.add_axes([0.33, y_pos, 0.25, 0.06])
        ax_max = fig_ctrl.add_axes([0.63, y_pos, 0.25, 0.06])
        
        # Formater les valeurs par défaut proprement (scientifique si nécessaire)
        init_min = f"{y_mins[i]:.4g}"
        init_max = f"{y_maxs[i]:.4g}"
        
        box_min = TextBox(ax_min, '', initial=init_min)
        box_max = TextBox(ax_max, '', initial=init_max)
        
        text_boxes_min.append(box_min)
        text_boxes_max.append(box_max)

        # Création d'une fonction de mise à jour spécifique à la ligne i (liaison tardive via l'argument par défaut)
        def make_update_handler(row_index):
            def update_y_axis(text):
                try:
                    val_min = float(text_boxes_min[row_index].text)
                    val_max = float(text_boxes_max[row_index].text)
                    
                    # On applique le changement uniquement sur le sous-graphique concerné
                    axes[row_index].set_ylim(val_min, val_max)
                    fig.canvas.draw_idle()
                except ValueError:
                    print(f"Erreur : Valeur invalide saisie pour {row_names[row_index]}")
            return update_y_axis

        handler = make_update_handler(i)
        box_min.on_submit(handler)
        box_max.on_submit(handler)

    # ==========================================================
    # Graphique Fit (inchangé)
    # ==========================================================
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