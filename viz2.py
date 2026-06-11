import os
import struct
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION EXACTE ---
NSubFaults = 450
BYTES_PER_PARAM = 96
step_size_bytes = 8 + (NSubFaults * BYTES_PER_PARAM)

bin_files = sorted(glob.glob("chain_cold_*.bin"))
if not bin_files:
    print("Erreur : Aucun fichier 'chain_cold_X.bin' trouvé.")
    exit()

num_chains = len(bin_files)

# =====================================================================
# FIGURE 1 : LOG-VRAISEMBLANCE
# =====================================================================
fig_llk, axes_llk = plt.subplots(num_chains, 1, figsize=(12, 2.5 * num_chains), sharex=False)
if num_chains == 1: axes_llk = [axes_llk]

models_data = {}
colors = plt.cm.tab10(np.linspace(0, 1, num_chains))
dt = np.dtype([('p1', '<f8'), ('p2', '<f8'), ('p3', '<f8'), ('p4', '<f8'), ('padding', 'V64')])

for idx, file_path in enumerate(bin_files):
    chain_idx = file_path.split("_")[-1].replace(".bin", "")
    num_steps = os.path.getsize(file_path) // step_size_bytes
    if num_steps == 0: continue
        
    llk_history = []
    first_model = last_model = None
    
    with open(file_path, "rb") as f:
        for step in range(num_steps):
            llk_bytes = f.read(8)
            if not llk_bytes: break
            llk_history.append(struct.unpack("d", llk_bytes)[0])
            
            model_bytes = f.read(NSubFaults * BYTES_PER_PARAM)
            if step == 0 or step == num_steps - 1:
                raw_data = np.frombuffer(model_bytes, dtype=dt)
                model_array = np.column_stack((raw_data['p1'], raw_data['p2'], raw_data['p3'], raw_data['p4']))
                if step == 0: first_model = model_array.copy()
                if step == num_steps - 1: last_model = model_array.copy()
                
    models_data[chain_idx] = (first_model, last_model)
    
    ax = axes_llk[idx]
    ax.plot(np.arange(len(llk_history)), llk_history, label=f"Chaîne {chain_idx}", 
            color=colors[idx], alpha=0.7, marker='.', linestyle='None', markersize=3)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylabel("LLK")
    ax.legend(loc="lower right")
    if idx == 0: ax.set_title("Évolution de la Log-Vraisemblance", fontweight='bold')

fig_llk.tight_layout()

# =====================================================================
# FIGURE 2 : PARAMÈTRES (MODÈLE INITIAL vs DERNIER MODÈLE PT)
# =====================================================================
try:
    df_true_params = pd.read_csv("model_parameters.csv")
    true_params_exist = True
except FileNotFoundError:
    true_params_exist = False

fig_params, axes_params = plt.subplots(len(models_data), 4, figsize=(16, 3 * len(models_data)), sharex='col')
if len(models_data) == 1: axes_params = np.expand_dims(axes_params, axis=0)
param_names = ["k_a_sigma", "b_a", "D_c_inv", "Dtau_asigma"]
x_faults = np.arange(NSubFaults)

for row_idx, (chain_idx, (mod_init, mod_final)) in enumerate(models_data.items()):
    for col_idx in range(4):
        ax = axes_params[row_idx, col_idx]
        
        # Vrai modèle cible en noir pointillé
        if true_params_exist:
            ax.plot(x_faults, df_true_params.iloc[:, col_idx+1], label="Init/Cible", color="black", linestyle="--", linewidth=1.5)
        
        # Dernier modèle trouvé par le PT en bleu
        ax.plot(x_faults, mod_final[:, col_idx], label="Dernier PT", color="blue", alpha=0.8)
        
        ax.grid(True, linestyle=":", alpha=0.5)
        if row_idx == 0: ax.set_title(param_names[col_idx], fontweight='bold')
        if col_idx == 0: ax.set_ylabel(f"Chaîne {chain_idx}")
        if row_idx == 0 and col_idx == 0: ax.legend(fontsize=8)

fig_params.tight_layout()

# =====================================================================
# FIGURE 3 : DÉPLACEMENTS (6 SOUS-GRAPHIQUES SÉPARÉS)
# =====================================================================
try:
    df_data = pd.read_csv("surface_responses.csv")
    df_pred = pd.read_csv("last_model_responses.csv")

    # Grille 3 Lignes (Composantes) x 2 Colonnes (Stations)
    fig_disp, axes_disp = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    
    components = ["North", "East", "Depth"]
    titles = ["Nord", "Est", "Profondeur (Vertical)"]
    
    # Couleurs distinctes pour éviter toute confusion
    color_target = "black"
    color_pred = "red"
    
    for st_idx in range(2): # Station 1 (colonne 0) et Station 2 (colonne 1)
        for comp_idx, comp in enumerate(components):
            ax = axes_disp[comp_idx, st_idx]
            col_name = f"St{st_idx+1}_{comp}"
            
            # Trace la Cible (Générée au début du main)
            ax.plot(df_data["Time"], df_data[col_name], label="Cible (Data)", color=color_target, linestyle="--", linewidth=2, alpha=0.7)
            # Trace la Prédiction (Générée à la fin du main via lecture du .bin)
            ax.plot(df_pred["Time"], df_pred[col_name], label="Prédiction PT", color=color_pred, linewidth=1.5)
            
            ax.grid(True, linestyle=":", alpha=0.6)
            
            if st_idx == 0: ax.set_ylabel("Déplacement (cm)")
            if comp_idx == 2: ax.set_xlabel("Temps (h)")
            
            if comp_idx == 0: 
                ax.set_title(f"STATION {st_idx+1} - {titles[comp_idx]}", fontweight='bold')
            else:
                ax.set_title(f"{titles[comp_idx]}")
                
            if comp_idx == 0 and st_idx == 0: ax.legend()

    fig_disp.tight_layout()

except FileNotFoundError:
    print("\nErreur : Assure-toi d'avoir compilé et relancé le code C++ pour générer les fichiers CSV.")

plt.show()