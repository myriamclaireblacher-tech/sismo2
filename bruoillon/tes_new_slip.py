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

NSubFaults = 16  # Nombre de sous-failles (Grille 4x4)
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

param1_inf = param1_min_possible
param1_sup = param1_max_possible

param2_inf = param2_min_possible
param2_sup = param2_max_possible

param3_inf = param3_min_possible
param3_sup = param3_max_possible

# --- VALEURS THÉORIQUES MIN ET MAX POSSIBLES (Issues de tes "Bounds_Param") ---
param1_min_possible = 0.0   
param1_max_possible = 5  

param2_min_possible = 0.0   
param2_max_possible = 5.0   

param3_min_possible = -2.0  
param3_max_possible = 2.0   

# =============================================================================
# 2. LECTURE ET FUSION DE TOUS LES FICHIERS
# =============================================================================
files_to_analyze = glob.glob(file_pattern)

if not files_to_analyze:
    raise FileNotFoundError(f"Aucun fichier correspondant à '{file_pattern}' n'a été trouvé dans le dossier actuel.")

print(f"Trouvé {len(files_to_analyze)} fichier(s) de chaîne froide à analyser.")

doubles_per_step = 1 + NSubFaults * 3
skip_iterations = 15000 * NSubFaults

all_p1_samples = []
all_p2_samples = []
all_p3_samples = []

for filepath in sorted(files_to_analyze):
    print(f"Traitement de {os.path.basename(filepath)}...")
    
    raw_data = np.fromfile(filepath, dtype=np.float64)
    total_steps = len(raw_data) // doubles_per_step
    
    if total_steps <= skip_iterations:
        print(f"  -> ATTENTION : Fichier ignoré (seulement {total_steps} itérations, "
              f"inférieur au burn-in de {skip_iterations}).")
        continue
        
    data_matrix = raw_data[:total_steps * doubles_per_step].reshape(total_steps, doubles_per_step)
    post_burn_data = data_matrix[skip_iterations:, 1:] 
    
    params_3d = post_burn_data.reshape(-1, NSubFaults, 3)
    
    all_p1_samples.append(params_3d[:, :, 0])
    all_p2_samples.append(params_3d[:, :, 1])
    all_p3_samples.append(params_3d[:, :, 2])

if not all_p1_samples:
    raise ValueError("Aucune donnée valide n'a pu être extraite après le burn-in.")

p1_data = np.vstack(all_p1_samples)
p2_data = np.vstack(all_p2_samples)
p3_data = np.vstack(all_p3_samples)

print(f"\n--- Statistique Globale ---")
print(f"Nombre total d'échantillons cumulés après burn-in : {p1_data.shape[0]}")

mean_p1 = np.mean(p1_data, axis=0).reshape(grid_size, grid_size)
mean_p2 = np.mean(p2_data, axis=0).reshape(grid_size, grid_size)
mean_p3 = np.mean(p3_data, axis=0).reshape(grid_size, grid_size)

# =============================================================================
# 3. TRACÉ DES HISTOGRAMMES FUSIONNÉS AVEC LIGNES CIBLES
# =============================================================================
param_names = ["a_sigma_k", "super_big_param", "big_param"]
param_datasets = [p1_data, p2_data, p3_data]
param_targets = [param1_target, param2_target, param3_target]

for p_idx, p_name in enumerate(param_names):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Histogrammes Cumulés - Paramètre : {p_name}", fontsize=15, fontweight='bold')
    
    current_data = param_datasets[p_idx]
    current_target = param_targets[p_idx]
    
    for sf in range(NSubFaults):
        row = sf // grid_size
        col = sf % grid_size
        ax = axes[row, col]
        
        # Histogramme des échantillons MCMC
        ax.hist(current_data[:, sf], bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax.set_title(f"Sous-faille {sf}", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Ajout de la ligne cible si elle est définie
        if current_target is not None:
            if isinstance(current_target, (list, np.ndarray, tuple)):
                tgt_value = current_target[sf]
            else:
                tgt_value = current_target # Valeur unique globale
                
            ax.axvline(x=tgt_value, color='red', linestyle='--', linewidth=2, label='Cible' if sf == 0 else "")
            if sf == 0:
                fig.legend(loc='upper right')
        
    plt.tight_layout()

# =============================================================================
# 4. TRACÉ DES GRILLES SPATIALES REPRÉSENTANT LES MOYENNES
# =============================================================================
fig_maps, axes_maps = plt.subplots(1, 3, figsize=(18, 6))
fig_maps.suptitle("Moyennes globales a posteriori sur la géométrie 4x4", fontsize=16, fontweight='bold')

maps_config = [
    {"mean_matrix": mean_p1, "name": "a_sigma_k", "vmin": param1_inf, "vmax": param1_sup, "p_min": param1_min_possible, "p_max": param1_max_possible, "cmap": "viridis"},
    {"mean_matrix": mean_p2, "name": "super_big_param", "vmin": param2_inf, "vmax": param2_sup, "p_min": param2_min_possible, "p_max": param2_max_possible, "cmap": "plasma"},
    {"mean_matrix": mean_p3, "name": "big_param", "vmin": param3_inf, "vmax": param3_sup, "p_min": param3_min_possible, "p_max": param3_max_possible, "cmap": "inferno"}
]

for idx, cfg in enumerate(maps_config):
    ax = axes_maps[idx]
    
    im = ax.imshow(cfg["mean_matrix"], cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"], origin='upper')
    
    for r in range(grid_size):
        for c in range(grid_size):
            sf_num = r * grid_size + c
            val_txt = f"{cfg['mean_matrix'][r, c]:.2f}"
            ax.text(c, r, f"SF {sf_num}\n{val_txt}", ha="center", va="center", 
                    color="white" if im.norm(cfg["mean_matrix"][r, c]) < 0.5 else "black", fontweight='bold', fontsize=9)
            
    ax.set_title(f"Paramètre : {cfg['name']}", fontsize=13, pad=10)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    
    fig_maps.colorbar(im, ax=ax, shrink=0.8)
    
    # Encadré des limites absolues min/max possibles
    ax.text(1.25, -0.08, f"Limites possibles :\nMin : {cfg['p_min']}\nMax : {cfg['p_max']}", 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', alpha=0.5))

plt.tight_layout()
print("\nAnalyse terminée. Affichage des graphiques...")
plt.show()