import matplotlib
matplotlib.use('TkAgg') 

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

# Valeurs cibles théoriques
k_asigma    = 0.01 / 0.4
b_a         = 0.17 / 0.4
D_c_inv     = 1.0 / 0.1
dtau_asigma = 2.0 / 0.4
V0__        = 0.08 * 100.0 / (365.0 * 24.0)
true_values = [k_asigma, b_a, D_c_inv, dtau_asigma, V0__]
viz1 = True

n_burn_phase = 500000

# CONFIGURATION DU FICHIER BINAIRE UNIQUE POUR LES TRACÉS ET HISTOGRAMMES
TARGET_BIN_FOR_HIST = "chain_cold_19.bin" 

# --- CONSTRUCTION AUTOMATIQUE DU DTYPE POUR 4 SOUS-FAILLES ---
fields = [('llk', '<f8')]
for sf in range(1, 5):
    fields.extend([
        (f'k_f{sf}', '<f8'), (f'a_sigma_f{sf}', '<f8'), (f'b_sigma_f{sf}', '<f8'), (f'D_c_f{sf}', '<f8'), 
        (f'Dtau_f{sf}', '<f8'), (f'V0__f{sf}', '<f8'), 
        (f'k_a_sigma_f{sf}', '<f8'), (f'b_a_f{sf}', '<f8'), (f'D_c_inv_f{sf}', '<f8'), (f'Dtau_asigma_f{sf}', '<f8'), 
        (f'coeff1_f{sf}', '<f8'), (f'coeff2_f{sf}', '<f8')
    ])
dt = np.dtype(fields)

def plot_all_results():
    """
    
    
    bin_files = sorted(glob.glob("chain_cold_*.bin"))
    
    if not bin_files:
        print("Erreur : Aucun fichier 'chain_cold_*.bin' trouvé.")
        return

    # ==========================================================
    # 1. TRACES TEMPORELLES (UNIQUEMENT POUR LE FICHIER SPÉCIFIÉ)
    # ==========================================================
    
    if viz1:
        print(f"Génération des séries temporelles MCMC pour {TARGET_BIN_FOR_HIST} uniquement...")
        # 6 lignes (LLK + 5 paramètres) x 4 colonnes (4 sous-failles)
        fig_trace, axes_trace = plt.subplots(6, 4, figsize=(20, 14), sharex=True)
        base_param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma", "V0_"]

        if os.path.exists(TARGET_BIN_FOR_HIST):
            data = np.fromfile(TARGET_BIN_FOR_HIST, dtype=dt)
            if len(data) > 0:
                steps = np.arange(len(data))
                llk = data['llk']
                
                # Tracé de la Log-Likelihood unique sur les 4 colonnes de sous-failles
                for col in range(4):
                    axes_trace[0, col].plot(steps, llk, color='royalblue', linewidth=1.5, zorder=10, label=TARGET_BIN_FOR_HIST if col==0 else "")
                
                # Tracé des 5 sous-paramètres pour les 4 sous-failles
                for sf_idx, sf_num in enumerate(range(1, 5)):
                    params_sf = [
                        data[f'k_a_sigma_f{sf_num}'], data[f'b_a_f{sf_num}'], 
                        data[f'D_c_inv_f{sf_num}'], data[f'Dtau_asigma_f{sf_num}'], data[f'V0__f{sf_num}']
                    ]
                    for i in range(5):
                        axes_trace[i+1, sf_idx].plot(steps, params_sf[i], color='royalblue', linewidth=1.2, zorder=10)
        else:
            print(f"[Alerte] Impossible de tracer les séries temporelles : {TARGET_BIN_FOR_HIST} introuvable.")

        # Habillage des axes
        for sf_idx, sf_num in enumerate(range(1, 5)):
            axes_trace[0, sf_idx].set_ylabel("Log-Likelihood", fontweight='bold')
            axes_trace[0, sf_idx].set_title(f"Séries MCMC - Sous-Faille {sf_num}", fontweight='bold', fontsize=12)
            if sf_idx == 0: axes_trace[0, sf_idx].legend(fontsize='small', loc='lower right')

            for i in range(5):
                ax = axes_trace[i+1, sf_idx]
                ax.set_ylabel(f"{base_param_names[i]} (F{sf_num})", fontweight='bold')
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.axhline(y=true_values[i], color='black', linestyle='--', linewidth=1.8)

            axes_trace[-1, sf_idx].set_xlabel("Itérations (Pas MCMC)", fontweight='bold')
            
        fig_trace.tight_layout()
        
    # ==========================================================
    # 2. CORNER PLOT (UNIQUEMENT SUR LE FICHIER BIN SPÉCIFIÉ)
    # ==========================================================
    
    if os.path.exists(TARGET_BIN_FOR_HIST):
        print(f"\nLecture de {TARGET_BIN_FOR_HIST} pour la construction des histogrammes...")
        data_cold = np.fromfile(TARGET_BIN_FOR_HIST, dtype=dt)
        n_steps_cold = len(data_cold)
        
        if n_steps_cold > 0:
            # Recherche de l'index optimal (Maximum de Vraisemblance)
            best_idx = np.argmax(data_cold['llk'])
            best_llk_val = data_cold['llk'][best_idx]
            print(f"[{TARGET_BIN_FOR_HIST}] Meilleur LLK localisé à l'index {best_idx} : {best_llk_val}")

            # Ajustement automatique du Burn-in si la chaîne est trop courte
            burn_idx = n_steps_cold // 2 if n_burn_phase >= n_steps_cold else n_burn_phase

            # Génération d'un Corner Plot complet (5x5) pour chacune des 4 sous-failles
            for sf in range(1, 5):
                print(f"Génération du Corner Plot : Sous-faille {sf}/4...")
                
                # Extraction et filtrage post burn-in
                k_a_sigma_sf = data_cold[f'k_a_sigma_f{sf}'][burn_idx:]
                b_a_sf       = data_cold[f'b_a_f{sf}'][burn_idx:]
                Dc_inv_raw   = data_cold[f'D_c_inv_f{sf}'][burn_idx:]
                Dtau_as_sf   = data_cold[f'Dtau_asigma_f{sf}'][burn_idx:]
                V0__sf       = data_cold[f'V0__f{sf}'][burn_idx:]

                # Transformation logarithmique de la variable inverse
                safe_Dc_inv = np.where(Dc_inv_raw > 0, Dc_inv_raw, 1e-12)
                params_cold = [k_a_sigma_sf, b_a_sf, np.log10(safe_Dc_inv), Dtau_as_sf, V0__sf]
                
                # Extraction de la position du Meilleur LLK
                best_values = [
                    data_cold[f'k_a_sigma_f{sf}'][best_idx],
                    data_cold[f'b_a_f{sf}'][best_idx],
                    np.log10(data_cold[f'D_c_inv_f{sf}'][best_idx]) if data_cold[f'D_c_inv_f{sf}'][best_idx] > 0 else -12,
                    data_cold[f'Dtau_asigma_f{sf}'][best_idx],
                    data_cold[f'V0__f{sf}'][best_idx]
                ]

                corner_true_values = [k_asigma, b_a, np.log10(D_c_inv), dtau_asigma, V0__]
                corner_param_names = [f"k_a_sigma (F{sf})", f"b_a (F{sf})", f"log10(D_c_inv) (F{sf})", f"Dtau_asigma (F{sf})", f"V0_ (F{sf})"]

                fig_corner, axes_corner = plt.subplots(5, 5, figsize=(14, 14))
                fig_corner.suptitle(f"F{sf} ({TARGET_BIN_FOR_HIST}) - Échantillons a posteriori (Burn-in : {burn_idx})", fontweight='bold', fontsize=14)

                # Synchronisation 2D matricielle des axes
                for i in range(5):
                    for j in range(5):
                        if i >= j:
                            if i != j: axes_corner[i, j].sharex(axes_corner[j, j])
                            if i > j and j > 0: axes_corner[i, j].sharey(axes_corner[i, 0])

                # Remplissage du Corner Plot
                for i in range(5):
                    for j in range(5):
                        ax = axes_corner[i, j]
                        if i == j:  # Diagonale : Marginale 1D (Histogramme)
                            ax.hist(params_cold[i], bins=40, color='royalblue', edgecolor='black', alpha=0.7, density=True)
                            ax.axvline(corner_true_values[i], color='red', linestyle='--', linewidth=2.0, label="Cible")
                            ax.axvline(best_values[i], color='darkviolet', linestyle=':', linewidth=2.0, label="Meilleur LLK")
                            if i == 0: ax.legend(fontsize='x-small')
                        elif i > j: # Triangle inférieur : Jointes 2D (Histogramme bidimensionnel)
                            ax.hist2d(params_cold[j], params_cold[i], bins=35, cmap='Blues', cmin=1)
                            ax.plot(corner_true_values[j], corner_true_values[i], marker='+', color='red', markersize=8, markeredgewidth=1.5)
                            ax.plot(best_values[j], best_values[i], marker='x', color='darkviolet', markersize=8, markeredgewidth=1.5)
                        else:       # Triangle supérieur : Vide
                            ax.axis('off')
                            
                        # Formatage des labels externes
                        if i == 4: ax.set_xlabel(corner_param_names[j], fontweight='bold', fontsize=9)
                        if j == 0 and i > 0: ax.set_ylabel(corner_param_names[i], fontweight='bold', fontsize=9)

                        # Invisibilité des axes intérieurs redondants
                        if i < 4: ax.tick_params(labelbottom=False)
                        if j > 0 or i == 0: ax.tick_params(labelleft=False)

                fig_corner.subplots_adjust(left=0.08, bottom=0.08, right=0.96, top=0.92, wspace=0.18, hspace=0.15)
                output_image = f"corner_plot_faille_{sf}.png"
                fig_corner.savefig(output_image, dpi=200, bbox_inches='tight')
                plt.close(fig_corner)  
                print(f"  -> Sauvegarde : {output_image}")
    else:
        print(f"\n[Alerte] Le fichier spécifié '{TARGET_BIN_FOR_HIST}' n'existe pas. Pas d'histogrammes générés.")
    """

    # ==========================================================
    # 3. GRAPHIQUE FIT (STATIONS / COMPOSANTES NORD-EST-Z)
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
            fig_disp, axes_disp = plt.subplots(3, num_stations, figsize=(4 * num_stations, 8), sharex=True, squeeze=False)
            
            components = ["North", "East", "Depth"]
            titles = ["Nord", "Est", "Profondeur"]
            
            for s_idx, station in enumerate(stations):
                for c_idx, comp in enumerate(components):
                    ax = axes_disp[c_idx, s_idx]
                    col_name = f"{station}_{comp}"
                    
                    if col_name in df_data.columns and col_name in df_pred.columns:
                        ax.plot(df_data["Time"], df_data[col_name], 'k--', label="Données (Cible)", linewidth=1.8, alpha=0.8)
                        ax.plot(df_pred["Time"], df_pred[col_name], 'r-', label="Modèle inversé", linewidth=1.3)
                    
                    ax.grid(True, linestyle=":", alpha=0.5)
                    if c_idx == 0: ax.set_title(f"{station}\n{titles[c_idx]}", fontweight='bold', fontsize=11)
                    else: ax.set_title(titles[c_idx], fontsize=10)
                        
                    if s_idx == 0 and c_idx == 1: ax.set_ylabel("Déplacement de surface (cm)", fontweight='bold')
                    if c_idx == 2: ax.set_xlabel("Temps (heures)", fontweight='bold')
                    if c_idx == 0 and s_idx == 0: ax.legend(fontsize='small')

            fig_disp.tight_layout()
            print("\nGraphique de fit des stations généré avec succès !")

    print("\nTous les traitements sont finis. Affichage des fenêtres à l'écran...")
    
    plt.show()
    

if __name__ == "__main__":
    plot_all_results()