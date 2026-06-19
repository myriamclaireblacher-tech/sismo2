import matplotlib
matplotlib.use('TkAgg') 

import numpy as np
import matplotlib.pyplot as plt
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

n_burn_phase = 2000000

# Mappage de la mémoire C++ (1 LLK + 12 variables de la classe Param)
dt = np.dtype([
    ('llk', '<f8'),
    ('k', '<f8'), ('a_sigma', '<f8'), ('b_sigma', '<f8'), ('D_c', '<f8'), 
    ('Dtau', '<f8'), ('V0_', '<f8'), 
    ('k_a_sigma', '<f8'), ('b_a', '<f8'), ('D_c_inv', '<f8'), ('Dtau_asigma', '<f8'), 
    ('coeff1', '<f8'), ('coeff2', '<f8')
])

def plot_histograms_and_fit():
    # ==========================================================
    # 1. CORNER PLOT (CLASSIQUE ET RAPIDE)
    # ==========================================================
    filename_cold = "chain_cold_0.bin"
    if os.path.exists(filename_cold):
        print("Lecture des données MCMC...")
        data_cold = np.fromfile(filename_cold, dtype=dt)
        n_steps_cold = len(data_cold)
        
        if n_steps_cold > 0:
            burn_idx = n_steps_cold // 2 if n_burn_phase >= n_steps_cold else n_burn_phase

            # Sécurité pour le log10 (évite les valeurs négatives ou nulles)
            safe_Dc_inv = np.where(data_cold['D_c_inv'][burn_idx:] > 0, data_cold['D_c_inv'][burn_idx:], 1e-12)

            params_cold = [
                data_cold['k_a_sigma'][burn_idx:],           
                data_cold['b_a'][burn_idx:],           
                np.log10(safe_Dc_inv), 
                data_cold['Dtau_asigma'][burn_idx:],          
                data_cold['V0_'][burn_idx:]            
            ]

            corner_param_names = ["k_a_sigma", "b_a", "log10(D_c_inv)", "Dtau_asigma", "V0_"]
            corner_true_values = [k_asigma, b_a, np.log10(D_c_inv), dtau_asigma, V0__]

            fig_corner, axes_corner = plt.subplots(5, 5, figsize=(15, 15))
            fig_corner.suptitle(f"Distributions a posteriori (Burn-in : {burn_idx})", fontweight='bold', fontsize=16)

            # --- SYNCHRONISATION DES AXES ---
            for i in range(5):
                for j in range(5):
                    if i >= j:
                        if i != j: axes_corner[i, j].sharex(axes_corner[j, j])
                        if i > j and j > 0: axes_corner[i, j].sharey(axes_corner[i, 0])

            print("Génération du Corner Plot...")
            # --- DESSIN DES HISTOGRAMMES ---
            for i in range(5):
                for j in range(5):
                    ax = axes_corner[i, j]
                    
                    if i == j: # DIAGONALE (Histogramme 1D)
                        ax.hist(params_cold[i], bins=50, color='royalblue', edgecolor='black', alpha=0.7, density=True)
                        ax.axvline(corner_true_values[i], color='red', linestyle='--', linewidth=2.5, label="Cible")
                        if i == 0: ax.legend(fontsize='small')
                        
                    elif i > j: # GRAPHIQUES 2D (hist2d)
                        ax.hist2d(params_cold[j], params_cold[i], bins=40, cmap='Blues', cmin=1)
                        ax.plot(corner_true_values[j], corner_true_values[i], marker='+', color='red', markersize=10, markeredgewidth=2)

                    else: # TRIANGLE SUPÉRIEUR VIDE
                        ax.axis('off')
                        
                    # --- AFFICHAGE DES NOMS DES PARAMÈTRES ---
                    if i == 4: 
                        ax.set_xlabel(corner_param_names[j], fontweight='bold')
                    else:
                        if i >= j: ax.set_xticklabels([]) 
                        
                    if j == 0 and i > 0: 
                        ax.set_ylabel(corner_param_names[i], fontweight='bold')
                    else:
                        if i >= j: ax.set_yticklabels([]) 

            plt.tight_layout()
            fig_corner.subplots_adjust(top=0.94)
    else:
        print(f"Erreur : Fichier '{filename_cold}' introuvable.")


    # ==========================================================
    # 2. GRAPHIQUE FIT (TOUTES STATIONS / COMPOSANTES)
    # ==========================================================
    if os.path.exists("surface_responses.csv") and os.path.exists("final_results.csv"):
        print("Génération du graphique de Fit des stations...")
        df_target = pd.read_csv("surface_responses.csv")
        df_pred = pd.read_csv("final_results.csv")

        station_cols = [c for c in df_target.columns if c.endswith('_North')]
        stations = [c.split('_')[0] for c in station_cols]
        num_stations = len(stations)

        if num_stations > 0:
            fig_disp, axes_disp = plt.subplots(3, num_stations, figsize=(5 * num_stations, 9), sharex=True)
            if num_stations == 1: axes_disp = axes_disp.reshape(3, 1)

            components = ["North", "East", "Depth"]
            titles = ["Nord", "Est", "Profondeur"]
            
            for s_idx, station in enumerate(stations):
                for c_idx, comp in enumerate(components):
                    ax = axes_disp[c_idx, s_idx]
                    col_name = f"{station}_{comp}"
                    
                    if col_name in df_target.columns and col_name in df_pred.columns:
                        ax.plot(df_target["Time"], df_target[col_name], 'k--', label="Cible (Data)", linewidth=2, alpha=0.7)
                        ax.plot(df_pred["Time"], df_pred[col_name], 'r-', label="Modèle Final", linewidth=1.5)
                    
                    ax.grid(True, linestyle=":", alpha=0.6)
                    
                    if c_idx == 0: ax.set_title(f"{station} - {titles[c_idx]}", fontweight='bold')
                    else: ax.set_title(f"{titles[c_idx]}")
                        
                    if s_idx == 0 and c_idx == 1: ax.set_ylabel("Déplacement (cm)")
                    if c_idx == 2: ax.set_xlabel("Temps (h)")
                    if c_idx == 0 and s_idx == 0: ax.legend()

            fig_disp.tight_layout()

    print("Terminé ! Affichage des fenêtres.")
    plt.show()

plot_histograms_and_fit()