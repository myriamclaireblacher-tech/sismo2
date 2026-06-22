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
viz1 = True

n_burn_phase = 200000

# Mappage de la mémoire C++ pour 2 sous-failles (1 LLK + 12 variables par sous-faille)
dt = np.dtype([
    ('llk', '<f8'),
    
    # SOUS-FAILLE 1
    ('k_f1', '<f8'), ('a_sigma_f1', '<f8'), ('b_sigma_f1', '<f8'), ('D_c_f1', '<f8'), 
    ('Dtau_f1', '<f8'), ('V0__f1', '<f8'), 
    ('k_a_sigma_f1', '<f8'), ('b_a_f1', '<f8'), ('D_c_inv_f1', '<f8'), ('Dtau_asigma_f1', '<f8'), 
    ('coeff1_f1', '<f8'), ('coeff2_f1', '<f8'),
    
    # SOUS-FAILLE 2
    ('k_f2', '<f8'), ('a_sigma_f2', '<f8'), ('b_sigma_f2', '<f8'), ('D_c_f2', '<f8'), 
    ('Dtau_f2', '<f8'), ('V0__f2', '<f8'), 
    ('k_a_sigma_f2', '<f8'), ('b_a_f2', '<f8'), ('D_c_inv_f2', '<f8'), ('Dtau_asigma_f2', '<f8'), 
    ('coeff1_f2', '<f8'), ('coeff2_f2', '<f8')
])

def plot_all_results():
    bin_files = sorted(glob.glob("chain_cold_*.bin"))
    
    if not bin_files:
        print("Erreur : Aucun fichier 'chain_cold_*.bin' trouvé.")
        return

    # ==========================================================
    # 1. TRACES TEMPORELLES (SÉRIES MCMC + LLK)
    # ==========================================================
    
    if viz1:
        # On crée une figure 6 lignes x 2 colonnes (une colonne par sous-faille)
        fig_trace, axes_trace = plt.subplots(6, 2, figsize=(16, 14), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(bin_files)))
        base_param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]

        for idx, filename in enumerate(bin_files):
            data = np.fromfile(filename, dtype=dt)
            if len(data) == 0: continue
            
            steps = np.arange(len(data))
            llk = data['llk']
            
            is_chain_zero = "chain_cold_0" in filename
            alpha_val = 1.0 if is_chain_zero else 0.5
            line_width = 1.5 if is_chain_zero else 0.8
            z_order = 10 if is_chain_zero else 1
            
            # Tracé du LLK (identique pour les deux colonnes)
            for col in [0, 1]:
                axes_trace[0, col].plot(steps, llk, color=colors[idx], alpha=alpha_val, linewidth=line_width, zorder=z_order, label=filename if col==0 else "")
            
            # Tracé des paramètres pour chaque sous-faille
            for sf_idx, sf_num in enumerate([1, 2]):
                params_sf = [
                    data[f'k_a_sigma_f{sf_num}'], data[f'b_a_f{sf_num}'], 
                    data[f'D_c_inv_f{sf_num}'], data[f'Dtau_asigma_f{sf_num}'], data[f'V0__f{sf_num}']
                ]
                for i in range(5):
                    axes_trace[i+1, sf_idx].plot(steps, params_sf[i], color=colors[idx], alpha=alpha_val, linewidth=line_width, zorder=z_order)

        # Mise en forme Traces
        for sf_idx, sf_num in enumerate([1, 2]):
            axes_trace[0, sf_idx].set_ylabel("Log-Likelihood", fontweight='bold')
            axes_trace[0, sf_idx].set_title(f"Traces MCMC - Sous-Faille {sf_num}", fontweight='bold', fontsize=14)
            if sf_idx == 0: axes_trace[0, sf_idx].legend(fontsize='small', loc='best')

            for i in range(5):
                ax = axes_trace[i+1, sf_idx]
                ax.set_ylabel(f"{base_param_names[i]} (F{sf_num})", fontweight='bold')
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.axhline(y=true_values[i], color='black', linestyle='--', linewidth=2.0, label="Cible")
                if i == 0 and sf_idx == 0: ax.legend(fontsize='small')

            axes_trace[-1, sf_idx].set_xlabel("Pas MCMC (Itérations acceptées)", fontweight='bold')
            
        fig_trace.tight_layout()
        

    # ==========================================================
    # 2. CORNER PLOT (1 FIGURE DISTINCTE PAR SOUS-FAILLE)
    # ==========================================================
    filename_cold = "chain_cold_0.bin"
    if os.path.exists(filename_cold):
        data_cold = np.fromfile(filename_cold, dtype=dt)
        n_steps_cold = len(data_cold)
        
        if n_steps_cold > 0:
            # Recherche de l'index ayant le meilleur Log-Likelihood (maximum absolu)
            best_idx = np.argmax(data_cold['llk'])
            best_llk_val = data_cold['llk'][best_idx]
            print(f"Meilleur LLK trouvé à l'itération {best_idx} : {best_llk_val}")

            burn_idx = n_steps_cold // 2 if n_burn_phase >= n_steps_cold else n_burn_phase

            # On boucle sur nos 2 sous-failles
            for sf in [1, 2]:
                print(f"Génération du Corner Plot pour la sous-faille {sf}...")
                
                # 1. Extraction des données pour les histogrammes (Après Burn-in)
                k_a_sigma_sf = data_cold[f'k_a_sigma_f{sf}'][burn_idx:]
                b_a_sf       = data_cold[f'b_a_f{sf}'][burn_idx:]
                Dc_inv_raw   = data_cold[f'D_c_inv_f{sf}'][burn_idx:]
                Dtau_as_sf   = data_cold[f'Dtau_asigma_f{sf}'][burn_idx:]
                V0__sf       = data_cold[f'V0__f{sf}'][burn_idx:]

                safe_Dc_inv = np.where(Dc_inv_raw > 0, Dc_inv_raw, 1e-12)
                params_cold = [k_a_sigma_sf, b_a_sf, np.log10(safe_Dc_inv), Dtau_as_sf, V0__sf]
                
                # 2. Extraction des valeurs uniques correspondant au Meilleur LLK
                best_k_a_sigma = data_cold[f'k_a_sigma_f{sf}'][best_idx]
                best_b_a       = data_cold[f'b_a_f{sf}'][best_idx]
                best_Dc_inv    = data_cold[f'D_c_inv_f{sf}'][best_idx]
                best_Dtau_as   = data_cold[f'Dtau_asigma_f{sf}'][best_idx]
                best_V0__      = data_cold[f'V0__f{sf}'][best_idx]
                
                safe_best_Dc_inv = np.log10(best_Dc_inv) if best_Dc_inv > 0 else -12
                corner_best_values = [best_k_a_sigma, best_b_a, safe_best_Dc_inv, best_Dtau_as, best_V0__]

                # Valeurs théoriques cibles
                corner_true_values = [k_asigma, b_a, np.log10(D_c_inv), dtau_asigma, V0__]
                corner_param_names = [f"k_a_sigma (F{sf})", f"b_a (F{sf})", f"log10(D_c_inv) (F{sf})", f"Dtau_asigma (F{sf})", f"V0_ (F{sf})"]

                fig_corner, axes_corner = plt.subplots(5, 5, figsize=(15, 15))
                fig_corner.suptitle(f"Distributions a posteriori - Sous-Faille {sf} (Burn-in : {burn_idx})", fontweight='bold', fontsize=16)

                # --- SYNCHRONISATION DES AXES ---
                for i in range(5):
                    for j in range(5):
                        if i >= j:
                            if i != j: axes_corner[i, j].sharex(axes_corner[j, j])
                            if i > j and j > 0: axes_corner[i, j].sharey(axes_corner[i, 0])

                # --- DESSIN DES HISTOGRAMMES ---
                for i in range(5):
                    for j in range(5):
                        ax = axes_corner[i, j]
                        if i == j: # DIAGONALE (Histogramme 1D)
                            ax.hist(params_cold[i], bins=50, color='royalblue', edgecolor='black', alpha=0.7, density=True)
                            ax.axvline(corner_true_values[i], color='red', linestyle='--', linewidth=2.5, label="Cible")
                            # Rajout de la ligne verticale violette pour le meilleur LLK
                            ax.axvline(corner_best_values[i], color='darkviolet', linestyle=':', linewidth=2.5, label="Meilleur LLK")
                            if i == 0: ax.legend(fontsize='small')
                        elif i > j: # GRAPHIQUES 2D
                            ax.hist2d(params_cold[j], params_cold[i], bins=40, cmap='Blues', cmin=1)
                            # Croix rouge pour la cible
                            ax.plot(corner_true_values[j], corner_true_values[i], marker='+', color='red', markersize=10, markeredgewidth=2)
                            # Croix violette pour le meilleur LLK
                            ax.plot(corner_best_values[j], corner_best_values[i], marker='x', color='darkviolet', markersize=10, markeredgewidth=2)
                        else: # TRIANGLE SUPÉRIEUR VIDE
                            ax.axis('off')
                            
                        # --- AFFICHAGE ROBUSTE DES AXES SUR LES BORDS ---
                        if i == 4: ax.set_xlabel(corner_param_names[j], fontweight='bold')
                        if j == 0 and i > 0: ax.set_ylabel(corner_param_names[i], fontweight='bold')

                        # Gestion de la visibilité des graduations pour éviter les axes vides
                        if i < 4:
                            ax.tick_params(labelbottom=False)
                        if j > 0 or i == 0:
                            ax.tick_params(labelleft=False)

                # Marges de sécurité fixes pour garantir l'affichage des valeurs
                fig_corner.subplots_adjust(left=0.08, bottom=0.08, right=0.96, top=0.92, wspace=0.25, hspace=0.25)
                
                # Enregistrement automatique de la figure courante
                output_image = f"corner_plot_faille_{sf}.png"
                fig_corner.savefig(output_image, dpi=300, bbox_inches='tight')
                print(f"-> Figure de la sous-faille {sf} enregistrée avec succès sous : {output_image}")

    # ==========================================================
    # 3. GRAPHIQUE FIT (STATIONS / COMPOSANTES)
    # ==========================================================
    file_data = "surface_responses.csv"
    file_pred = "best_results.csv"

    if os.path.exists(file_data) and os.path.exists(file_pred):
        df_data = pd.read_csv(file_data)
        df_pred = pd.read_csv(file_pred)
        
        df_data.columns = df_data.columns.str.strip()
        df_pred.columns = df_pred.columns.str.strip()

        stations = [col.split("_North")[0] for col in df_data.columns if col.endswith("_North")]
        
        if len(stations) > 0:
            num_stations = len(stations)
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

    print("\nTerminé ! Affichage des fenêtres à l'écran...")
    plt.show()

if __name__ == "__main__":
    plot_all_results()