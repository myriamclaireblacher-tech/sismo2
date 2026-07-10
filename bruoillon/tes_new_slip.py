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
    
    


import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION, LIMITES ET VALEURS CIBLES (À REMPLIR)
# =============================================================================
file_pattern = "Easy_chain_cold_*.bin"  

NSubFaults = 32  # Nombre de sous-failles (Grille 4x4)
grid_size = 4    # 4x4

# --- EMPLACEMENTS VIDES POUR LES VALEURS CIBLES (TRUE VALUES) ---
# Tu peux mettre soit un seul nombre : 2.5
# Soit une liste de 16 nombres : [1.0, 1.2, 1.5, ..., 2.0]
param1_target = 0.4  # Cible pour a_sigma_k
param2_target = np.exp(2/0.4)* 1  # Cible pour super_big_param
param3_target = 0.08/0.4  # Cible pour big_param

# --- EMPLACEMENTS VIDES POUR CONFIGURER LES BORNES DE L'ÉCHELLE DE COULEUR ---
param1_min_possible = 0.0   
param1_max_possible = 5  

param2_min_possible = 0.0   
param2_max_possible = 5.0   

param3_min_possible = -2.0  
param3_max_possible = 2.0   

param1_inf = 0
param1_sup = 5

param2_inf = 0
param2_sup = 1000

param3_inf = 0
param3_sup = 1

# --- VALEURS THÉORIQUES MIN ET MAX POSSIBLES (Issues de tes "Bounds_Param") ---
param1_min_possible = 0.0   
param1_max_possible = 5  

param2_min_possible = 0.0   
param2_max_possible = param2_sup   

param3_min_possible = param3_inf
param3_max_possible = param3_sup   

# --- EMPLACEMENTS VIDES POUR AJUSTER LES BORNES DES HISTOGRAMMES 1D (AXE X) ---
param1_hist_min, param1_hist_max = param1_min_possible, param1_max_possible
param2_hist_min, param2_hist_max = param2_min_possible, param2_max_possible
param3_hist_min, param3_hist_max = param3_min_possible, param3_max_possible


# =============================================================================
# 2. LECTURE, SELECTION DU MEILLEUR MODÈLE (MAX LLK) ET FUSION
# =============================================================================
files_to_analyze = glob.glob(file_pattern)

if not files_to_analyze:
    raise FileNotFoundError(f"Aucun fichier correspondant à '{file_pattern}' n'a été trouvé.")

print(f"Trouvé {len(files_to_analyze)} fichier(s) de chaîne froide à analyser.")

doubles_per_step = 1 + NSubFaults * 3
skip_iterations = 15000 * NSubFaults

all_llks = []
all_params_flat = []

for filepath in sorted(files_to_analyze):
    print(f"Traitement de {os.path.basename(filepath)}...")
    raw_data = np.fromfile(filepath, dtype=np.float64)
    total_steps = len(raw_data) // doubles_per_step
    
    if total_steps <= skip_iterations:
        print(f"  -> ATTENTION : Fichier ignoré (Burn-in non dépassé).")
        continue
        
    data_matrix = raw_data[:total_steps * doubles_per_step].reshape(total_steps, doubles_per_step)
    
    all_llks.append(data_matrix[skip_iterations:, 0])
    all_params_flat.append(data_matrix[skip_iterations:, 1:])

if not all_llks:
    raise ValueError("Aucune donnée valide n'a pu être extraite après le burn-in.")

global_llk = np.concatenate(all_llks)
global_params_flat = np.vstack(all_params_flat)

print(f"\n--- Statistique Globale ---")
print(f"Nombre total d'échantillons cumulés : {global_llk.shape[0]}")

# EXTRACTION DU MEILLEUR MODÈLE (MAX LLK)
best_step_idx = np.argmax(global_llk)
best_llk_value = global_llk[best_step_idx]
print(f"Meilleure Vraisemblance (Max LLK) trouvée : {best_llk_value}")

best_model_raw = global_params_flat[best_step_idx].reshape(NSubFaults, 3)
best_grid_p1 = best_model_raw[:, 0].reshape(grid_size, grid_size)
best_grid_p2 = best_model_raw[:, 1].reshape(grid_size, grid_size)
best_grid_p3 = best_model_raw[:, 2].reshape(grid_size, grid_size)

# Séparation de l'historique complet pour les plots
params_3d = global_params_flat.reshape(-1, NSubFaults, 3)
p1_hist_data = params_3d[:, :, 0]
p2_hist_data = params_3d[:, :, 1]
p3_hist_data = params_3d[:, :, 2]

# Helper interne pour extraire proprement les cibles scalaires ou vectorielles
def get_target_val(target, sf_idx):
    if target is None: return None
    return target[sf_idx] if isinstance(target, (list, np.ndarray, tuple)) else target

# =============================================================================
# 3. TRACÉ DES HISTOGRAMMES 1D FUSIONNÉS (SANS SHAREX MANDATOIRE)
# =============================================================================
param_names = ["a_sigma_k", "super_big_param", "big_param"]
param_datasets = [p1_hist_data, p2_hist_data, p3_hist_data]
param_targets = [param1_target, param2_target, param3_target]

hist_bounds = [
    {"min": param1_hist_min, "max": param1_hist_max},
    {"min": param2_hist_min, "max": param2_hist_max},
    {"min": param3_hist_min, "max": param3_hist_max}
]

for p_idx, p_name in enumerate(param_names):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(11, 11), sharex=False)
    fig.suptitle(f"Histogrammes de Densité 1D - Paramètre : {p_name}", fontsize=14, fontweight='bold')
    
    current_data = param_datasets[p_idx]
    current_bounds = hist_bounds[p_idx]
    
    for sf in range(NSubFaults):
        row, col = sf // grid_size, sf % grid_size
        ax = axes[row, col]
        
        ax.hist(current_data[:, sf], bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax.set_title(f"Sous-faille {sf}", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        if current_bounds["min"] is not None or current_bounds["max"] is not None:
            ax.set_xlim(current_bounds["min"], current_bounds["max"])
            
        tgt = get_target_val(param_targets[p_idx], sf)
        if tgt is not None:
            ax.axvline(x=tgt, color='red', linestyle='--', linewidth=1.5, label='Cible' if sf == 0 else "")
            if sf == 0: fig.legend(loc='upper right')
        
    plt.tight_layout()

# =============================================================================
# 4. CORNER PLOTS (HISTOGRAMMES 1D + 2D) POUR 3 SOUS-FAILLES ALÉATOIRES
# =============================================================================
np.random.seed(None) 
selected_subfaults = sorted(np.random.choice(NSubFaults, size=3, replace=False))
print(f"\nSous-failles sélectionnées aléatoirement pour les plots 2D : {selected_subfaults}")

for sf in selected_subfaults:
    print(f"Génération du Corner Plot : Sous-faille {sf}...")
    
    # Données pour la sous-faille courante
    params_cold = [p1_hist_data[:, sf], p2_hist_data[:, sf], p3_hist_data[:, sf]]
    
    # Meilleur modèle localisé
    best_values = [
        best_grid_p1[sf // grid_size, sf % grid_size],
        best_grid_p2[sf // grid_size, sf % grid_size],
        best_grid_p3[sf // grid_size, sf % grid_size]
    ]

    # Cibles
    corner_true_values = [
        get_target_val(param1_target, sf),
        get_target_val(param2_target, sf),
        get_target_val(param3_target, sf)
    ]
    corner_param_names = [f"a_sigma_k (SF{sf})", f"super_big_param (SF{sf})", f"big_param (SF{sf})"]

    fig_corner, axes_corner = plt.subplots(3, 3, figsize=(10, 10))
    fig_corner.suptitle(f"F{sf} - Corner Plot a posteriori (Max LLK: {best_llk_value:.2f})", fontweight='bold', fontsize=14)

    # Synchronisation 2D matricielle des axes
    for i in range(3):
        for j in range(3):
            if i >= j:
                if i != j: axes_corner[i, j].sharex(axes_corner[j, j])
                if i > j and j > 0: axes_corner[i, j].sharey(axes_corner[i, 0])

    # Remplissage du Corner Plot
    for i in range(3):
        for j in range(3):
            ax = axes_corner[i, j]
            if i == j:  # Diagonale : Marginale 1D
                ax.hist(params_cold[i], bins=40, color='royalblue', edgecolor='black', alpha=0.7, density=True)
                if corner_true_values[i] is not None:
                    ax.axvline(corner_true_values[i], color='red', linestyle='--', linewidth=2.0, label="Cible")
                ax.axvline(best_values[i], color='darkviolet', linestyle=':', linewidth=2.0, label="Meilleur LLK")
                if i == 0: ax.legend(fontsize='x-small')
                
            elif i > j: # Triangle inférieur : Histogramme 2D
                ax.hist2d(params_cold[j], params_cold[i], bins=35, cmap='Blues', cmin=1)
                if corner_true_values[j] is not None and corner_true_values[i] is not None:
                    ax.plot(corner_true_values[j], corner_true_values[i], marker='+', color='red', markersize=8, markeredgewidth=1.5)
                ax.plot(best_values[j], best_values[i], marker='x', color='darkviolet', markersize=8, markeredgewidth=1.5)
                
            else:       # Triangle supérieur : Vide
                ax.axis('off')
                
            # Formatage des labels (correction fontweight ici)
            if i == 2: ax.set_xlabel(corner_param_names[j], fontweight='bold', fontsize=10)
            if j == 0 and i > 0: ax.set_ylabel(corner_param_names[i], fontweight='bold', fontsize=10)

            # Invisibilité des axes intérieurs redondants
            if i < 2: ax.tick_params(labelbottom=False)
            if j > 0 or i == 0: ax.tick_params(labelleft=False)

    fig_corner.subplots
    
    
    plt.show()