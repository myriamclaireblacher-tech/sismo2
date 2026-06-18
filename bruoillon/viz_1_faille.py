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

    # ==========================================================
    # 1. CRÉATION DE LA FIGURE PRINCIPALE (TRACES)
    # ==========================================================
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(bin_files)))
    param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]
    
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
        
        y_mins[0] = min(y_mins[0], np.min(llk))
        y_maxs[0] = max(y_maxs[0], np.max(llk))
        
        axes[0].plot(steps, llk, color=colors[idx], linewidth=line_width, 
                     alpha=alpha_val, zorder=zorder, label=filename)
        
        for i in range(5):
            y_mins[i+1] = min(y_mins[i+1], np.min(params[i]))
            y_maxs[i+1] = max(y_maxs[i+1], np.max(params[i]))
            axes[i+1].plot(steps, params[i], color=colors[idx], linewidth=line_width, 
                           alpha=alpha_val, zorder=zorder)

    axes[0].set_ylabel("Log-Likelihood", fontweight='bold')
    axes[0].set_title("Superposition des traces MCMC", fontweight='bold', fontsize=14)
    axes[0].legend(fontsize='x-small', loc='best')

    for i in range(5):
        ax = axes[i+1]
        ax.set_ylabel(param_names[i], fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(y=true_values[i], color='black', linestyle='--', linewidth=1.5)
        if i == 0: ax.legend(fontsize='small')
    
    axes[-1].set_xlabel("Pas MCMC (ou lignes enregistrées)", fontweight='bold')
    plt.tight_layout()

    # ==========================================================
    # 2. PANNEAU DE CONTROLE DES AXES Y (inchangé)
    # ==========================================================
    fig_ctrl = plt.figure("Contrôle des Axes Y", figsize=(6, 8))
    fig_ctrl.text(0.5, 0.96, "Ajustement des échelles verticales (Y)", ha="center", va="top", fontweight="bold", fontsize=12)
    fig_ctrl.text(0.45, 0.91, "Y Min", ha="center", fontweight="bold")
    fig_ctrl.text(0.75, 0.91, "Y Max", ha="center", fontweight="bold")

    text_boxes_min = []
    text_boxes_max = []
    row_names = ["Log-Likelihood", "k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]

    for i in range(6):
        y_pos = 0.82 - (i * 0.13)
        fig_ctrl.text(0.05, y_pos + 0.03, row_names[i], va="center", fontweight="bold", fontsize=10)
        
        ax_min = fig_ctrl.add_axes([0.33, y_pos, 0.25, 0.06])
        ax_max = fig_ctrl.add_axes([0.63, y_pos, 0.25, 0.06])
        
        init_min = f"{y_mins[i]:.4g}" if y_mins[i] != float('inf') else "0"
        init_max = f"{y_maxs[i]:.4g}" if y_maxs[i] != float('-inf') else "0"
        
        box_min = TextBox(ax_min, '', initial=init_min)
        box_max = TextBox(ax_max, '', initial=init_max)
        
        text_boxes_min.append(box_min)
        text_boxes_max.append(box_max)

        def make_update_handler(row_index):
            def update_y_axis(text):
                try:
                    val_min = float(text_boxes_min[row_index].text)
                    val_max = float(text_boxes_max[row_index].text)
                    axes[row_index].set_ylim(val_min, val_max)
                    fig.canvas.draw_idle()
                except ValueError:
                    print(f"Erreur : Valeur invalide saisie pour {row_names[row_index]}")
            return update_y_axis

        handler = make_update_handler(i)
        box_min.on_submit(handler)
        box_max.on_submit(handler)

    # ==========================================================
    # 3. GRAPHIQUE FIT (Cible vs Prédiction)
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

    # ==========================================================
    # 4. CORNER PLOT AVEC HISTOGRAMMES DYNAMIQUES
    # ==========================================================
    filename_cold = "test_chain_cold_0.bin"
    if os.path.exists(filename_cold):
        data_cold = np.fromfile(filename_cold, dtype=np.float64)
        n_steps_cold = len(data_cold) // 13
        
        if n_steps_cold > 0:
            data_cold = data_cold[:n_steps_cold * 13].reshape((n_steps_cold, 13))
            
            n_burn_phase = 100000 
            if n_burn_phase >= n_steps_cold:
                burn_idx = n_steps_cold // 2
            else:
                burn_idx = n_burn_phase

            # Extraction avec log10 pour D_c_inv
            params_cold = [
                data_cold[burn_idx:, 7],           # k_a_sigma
                data_cold[burn_idx:, 8],           # b_a
                np.log10(data_cold[burn_idx:, 9]), # log10(D_c_inv)
                data_cold[burn_idx:, 10],          # Dtau_asigma
                data_cold[burn_idx:, 6]            # V0_
            ]

            corner_param_names = ["k_a_sigma", "b_a", "log10(D_c_inv)", "Dtau_asigma", "V0_"]
            corner_true_values = [k_asigma, b_a, np.log10(D_c_inv), dtau_asigma, V0__]

            fig_corner, axes_corner = plt.subplots(5, 5, figsize=(15, 15))
            fig_corner.suptitle(f"Distributions a posteriori (Chaîne 0, Burn-in : {burn_idx} lignes)\nZoomer sur les diagonales pour recalculer les histogrammes !", 
                                fontweight='bold', fontsize=16)

            # --- FONCTION POUR HISTOGRAMME DYNAMIQUE ---
            def setup_dynamic_hist(ax, data, true_val, add_legend):
                def draw_hist(xmin, xmax):
                    # Supprime uniquement les barres d'histogramme (pas la ligne cible)
                    for p in reversed(ax.patches):
                        p.remove()
                    
                    mask = (data >= xmin) & (data <= xmax)
                    visible_data = data[mask]
                    
                    if len(visible_data) > 5:
                        ax.hist(visible_data, bins=50, range=(xmin, xmax), 
                                color='royalblue', edgecolor='black', alpha=0.7, density=True)
                    ax.set_xlim(xmin, xmax)

                def on_xlim_changed(axes_obj):
                    if getattr(axes_obj, '_is_updating', False): return
                    axes_obj._is_updating = True
                    xmin, xmax = axes_obj.get_xlim()
                    draw_hist(xmin, xmax)
                    axes_obj._is_updating = False

                # Initialisation
                xmin, xmax = np.min(data), np.max(data)
                draw_hist(xmin, xmax)
                ax.axvline(true_val, color='red', linestyle='--', linewidth=2.5, label="Cible")
                if add_legend: ax.legend(fontsize='small')
                
                # Connexion de l'événement de zoom
                ax.callbacks.connect('xlim_changed', on_xlim_changed)
            # -------------------------------------------

            for i in range(5):
                for j in range(5):
                    ax = axes_corner[i, j]
                    
                    if i == j:
                        # Diagonale : Application de la fonction dynamique
                        setup_dynamic_hist(ax, params_cold[i], corner_true_values[i], add_legend=(i==0))
                        
                    elif i > j:
                        # Triangle inférieur : Histogramme 2D
                        ax.hist2d(params_cold[j], params_cold[i], bins=40, cmap='Blues', cmin=1)
                        ax.plot(corner_true_values[j], corner_true_values[i], marker='+', color='red', markersize=10, markeredgewidth=2)
                        
                    else:
                        ax.axis('off')
                        
                    # Gestion des labels
                    if i == 4: ax.set_xlabel(corner_param_names[j], fontweight='bold')
                    else:
                        if i >= j: ax.set_xticklabels([]) 
                        
                    if j == 0 and i > 0: ax.set_ylabel(corner_param_names[i], fontweight='bold')
                    else:
                        if i >= j: ax.set_yticklabels([]) 

            plt.tight_layout()
            fig_corner.subplots_adjust(top=0.94)

    plt.show()

plot_all_mcmc_chains()