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
viz1= True

n_burn_phase = 200000

# Mappage de la mémoire C++ (1 LLK + 12 variables de la classe Param)
dt = np.dtype([
    ('llk', '<f8'),
    ('k', '<f8'), ('a_sigma', '<f8'), ('b_sigma', '<f8'), ('D_c', '<f8'), 
    ('Dtau', '<f8'), ('V0_', '<f8'), 
    ('k_a_sigma', '<f8'), ('b_a', '<f8'), ('D_c_inv', '<f8'), ('Dtau_asigma', '<f8'), 
    ('coeff1', '<f8'), ('coeff2', '<f8')
])

def plot_all_results():
    bin_files = sorted(glob.glob("chain_cold_*.bin"))
    
    if not bin_files:
        print("Erreur : Aucun fichier 'chain_cold_*.bin' trouvé.")
        return

    # ==========================================================
    # 1. TRACES TEMPORELLES (SÉRIES MCMC + LLK)
    # ==========================================================
    """
    if viz1 :
        fig_trace, axes_trace = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(bin_files)))
        param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]

        for idx, filename in enumerate(bin_files):
            data = np.fromfile(filename, dtype=dt)
            if len(data) == 0: continue
            
            steps = np.arange(len(data))
            llk = data['llk']
            params = [data['k_a_sigma'], data['b_a'], data['D_c_inv'], data['Dtau_asigma'], data['V0_']]
            
            is_chain_zero = "chain_cold_0" in filename
            alpha_val = 1.0 if is_chain_zero else 0.5
            line_width = 1.5 if is_chain_zero else 0.8
            z_order = 10 if is_chain_zero else 1
            
            axes_trace[0].plot(steps, llk, color=colors[idx], alpha=alpha_val, linewidth=line_width, zorder=z_order, label=filename)
            for i in range(5):
                axes_trace[i+1].plot(steps, params[i], color=colors[idx], alpha=alpha_val, linewidth=line_width, zorder=z_order)

        axes_trace[0].set_ylabel("Log-Likelihood", fontweight='bold')
        axes_trace[0].set_title("Traces MCMC : Évolution des paramètres", fontweight='bold', fontsize=14)
        axes_trace[0].legend(fontsize='small', loc='best')

        for i in range(5):
            axes_trace[i+1].set_ylabel(param_names[i], fontweight='bold')
            axes_trace[i+1].grid(True, linestyle=":", alpha=0.6)
            axes_trace[i+1].axhline(y=true_values[i], color='black', linestyle='--', linewidth=2.0, label="Cible")
            if i == 0: axes_trace[i+1].legend(fontsize='small')

        axes_trace[-1].set_xlabel("Pas MCMC (Itérations acceptées)", fontweight='bold')
        fig_trace.tight_layout()
    """
    # ==========================================================
    # 2. CORNER PLOT (HISTOGRAMMES 1D ET 2D)
    # ==========================================================
    filename_cold = "chain_cold_0.bin"
    if os.path.exists(filename_cold):
        data_cold = np.fromfile(filename_cold, dtype=dt)
        n_steps_cold = len(data_cold)
        
        if n_steps_cold > 0:
            burn_idx = n_steps_cold // 2 if n_burn_phase >= n_steps_cold else n_burn_phase
            safe_Dc_inv = np.where(data_cold['D_c_inv'][burn_idx:] > 0, data_cold['D_c_inv'][burn_idx:], 1e-12)
            
            params_cold = [
                data_cold['k_a_sigma'][burn_idx:], data_cold['b_a'][burn_idx:],           
                np.log10(safe_Dc_inv), data_cold['Dtau_asigma'][burn_idx:], data_cold['V0_'][burn_idx:]            
            ]
            corner_true_values = [k_asigma, b_a, np.log10(D_c_inv), dtau_asigma, V0__]
            corner_param_names = ["k_a_sigma", "b_a", "log10(D_c_inv)", "Dtau_asigma", "V0_"]

            fig_corner, axes_corner = plt.subplots(5, 5, figsize=(15, 15))
            fig_corner.suptitle(f"Distributions a posteriori (Burn-in : {burn_idx})", fontweight='bold', fontsize=16)

            for i in range(5):
                for j in range(5):
                    if i >= j:
                        if i != j: axes_corner[i, j].sharex(axes_corner[j, j])
                        if i > j and j > 0: axes_corner[i, j].sharey(axes_corner[i, 0])

            for i in range(5):
                for j in range(5):
                    ax = axes_corner[i, j]
                    if i == j:
                        ax.hist(params_cold[i], bins=50, color='royalblue', edgecolor='black', alpha=0.7, density=True)
                        ax.axvline(corner_true_values[i], color='red', linestyle='--', linewidth=2.5, label="Cible")
                        if i == 0: ax.legend(fontsize='small')
                    elif i > j:
                        ax.hist2d(params_cold[j], params_cold[i], bins=40, cmap='Blues', cmin=1)
                        ax.plot(corner_true_values[j], corner_true_values[i], marker='+', color='red', markersize=10, markeredgewidth=2)
                    else:
                        ax.axis('off')
                     
                        
                    # --- AFFICHAGE ROBUSTE DES AXES ET DES VALEURS ---
                    # 1. On place les titres (Labels) uniquement sur les bords
                    if i == 4: ax.set_xlabel(corner_param_names[j], fontweight='bold')
                    if j == 0 and i > 0: ax.set_ylabel(corner_param_names[i], fontweight='bold')

                    # 2. On gère l'affichage des CHIFFRES sans casser les axes partagés
                    if i < 4:
                        ax.tick_params(labelbottom=False) # Cache les chiffres de l'axe X à l'intérieur
                    if j > 0 or i == 0:
                        ax.tick_params(labelleft=False)   # Cache les chiffres de l'axe Y à l'intérieur et pour la diagonale    

            # À remplacer à la toute fin du bloc "2. CORNER PLOT" (juste avant la Section 3)
            #fig_corner.tight_layout()
            fig_corner.subplots_adjust(top=0.94)
            
            # --- CODE AJOUTÉ POUR L'ENREGISTREMENT ---
            output_image = "corner_plot_distributions.png"
            fig_corner.savefig(output_image, dpi=300, bbox_inches='tight')
            print(f"Figure du Corner Plot enregistrée avec succès sous : {output_image}")
            # -----------------------------------------

    # ==========================================================
    # 3. GRAPHIQUE FIT (STATIONS / COMPOSANTES) - NOMS CORRIGÉS
    # ==========================================================
    file_data = "surface_responses.csv"
    file_pred = "best_results.csv"  # <--- LE BON NOM EST ICI !

    if os.path.exists(file_data) and os.path.exists(file_pred):
        df_data = pd.read_csv(file_data)
        df_pred = pd.read_csv(file_pred)
        
        # Nettoyage des espaces invisibles des noms de colonnes
        df_data.columns = df_data.columns.str.strip()
        df_pred.columns = df_pred.columns.str.strip()

        # Détection des stations (St1, St2, etc.)
        stations = [col.split("_North")[0] for col in df_data.columns if col.endswith("_North")]
        
        if len(stations) > 0:
            num_stations = len(stations)
            
            # Squeeze=False permet de garder un tableau 2D même s'il n'y a qu'une station
            fig_disp, axes_disp = plt.subplots(3, num_stations, figsize=(5 * num_stations, 9), sharex=True, squeeze=False)
            
            components = ["North", "East", "Depth"]
            titles = ["Nord", "Est", "Profondeur"]
            
            for s_idx, station in enumerate(stations):
                for c_idx, comp in enumerate(components):
                    ax = axes_disp[c_idx, s_idx]
                    col_name = f"{station}_{comp}"
                    
                    if col_name in df_data.columns and col_name in df_pred.columns:
                        ax.plot(df_data["Time"], df_data[col_name], 'k--', label="Cible (Data)", linewidth=2, alpha=0.7)
                        ax.plot(df_pred["Time"], df_pred[col_name], 'r-', label="Modèle Final", linewidth=1.5)
                    
                    ax.grid(True, linestyle=":", alpha=0.6)
                    if c_idx == 0: ax.set_title(f"{station} - {titles[c_idx]}", fontweight='bold')
                    else: ax.set_title(f"{titles[c_idx]}")
                        
                    if s_idx == 0 and c_idx == 1: ax.set_ylabel("Déplacement (cm)")
                    if c_idx == 2: ax.set_xlabel("Temps (h)")
                    if c_idx == 0 and s_idx == 0: ax.legend()

            fig_disp.tight_layout()
            print("Graphique des stations généré avec succès !")
        else:
            print("Erreur : Aucune colonne de station trouvée dans les CSV.")
    else:
        print(f"ERREUR : Il manque le fichier {file_data} ou {file_pred} dans le dossier !")

    print("Terminé ! Affichage des fenêtres.")
    plt.show()

plot_all_results()