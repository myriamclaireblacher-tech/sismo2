import os
import struct
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION EXACTE ---
NSubFaults = 1
BYTES_PER_PARAM = 96
step_size_bytes = 8 + (NSubFaults * BYTES_PER_PARAM)

bin_files = sorted(glob.glob("chain_cold_*.bin"))
if not bin_files:
    print("Erreur : Aucun fichier 'chain_cold_X.bin' trouvé.")
    exit()

num_chains = len(bin_files)
colors = plt.cm.tab10(np.linspace(0, 1, num_chains))

# Mappage EXACT de la mémoire de ta classe Param C++ (12 doubles = 96 octets)
dt = np.dtype([
    ('k', '<f8'), ('a_sigma', '<f8'), ('b_sigma', '<f8'), ('D_c', '<f8'), 
    ('V0_', '<f8'), ('Dtau', '<f8'), 
    ('k_a_sigma', '<f8'), ('b_a', '<f8'), ('D_c_inv', '<f8'), ('Dtau_asigma', '<f8'), 
    ('coeff1', '<f8'), ('coeff2', '<f8')
])

# =====================================================================
# LECTURE UNIQUE DES DONNÉES
# =====================================================================
models_data = {}
param_history = {}
all_llk_histories = {}

for idx, file_path in enumerate(bin_files):
    chain_idx = file_path.split("_")[-1].replace(".bin", "")
    num_steps = os.path.getsize(file_path) // step_size_bytes
    if num_steps == 0: continue
        
    llk_history = np.zeros(num_steps)
    
    # Pré-allocation en utilisant les vrais noms des paramètres
    param_history[chain_idx] = {
        'k_a_sigma': np.zeros(num_steps),
        'b_a': np.zeros(num_steps),
        'D_c_inv': np.zeros(num_steps),
        'Dtau_asigma': np.zeros(num_steps)
    }
    
    first_model = last_model = None
    
    with open(file_path, "rb") as f:
        for step in range(num_steps):
            llk_bytes = f.read(8)
            if not llk_bytes: break
            llk_history[step] = struct.unpack("d", llk_bytes)[0]
            
            model_bytes = f.read(NSubFaults * BYTES_PER_PARAM)
            raw_data = np.frombuffer(model_bytes, dtype=dt)
            
            # Stockage de la trace MCMC
            param_history[chain_idx]['k_a_sigma'][step] = raw_data['k_a_sigma'][0]
            param_history[chain_idx]['b_a'][step] = raw_data['b_a'][0]
            param_history[chain_idx]['D_c_inv'][step] = raw_data['D_c_inv'][0]
            param_history[chain_idx]['Dtau_asigma'][step] = raw_data['Dtau_asigma'][0]
            
            # Conservation du premier et dernier modèle
            if step == 0 or step == num_steps - 1:
                model_array = np.column_stack((
                    raw_data['k_a_sigma'], raw_data['b_a'], 
                    raw_data['D_c_inv'], raw_data['Dtau_asigma']
                ))
                if step == 0: first_model = model_array.copy()
                if step == num_steps - 1: last_model = model_array.copy()
                
    models_data[chain_idx] = (first_model, last_model)
    all_llk_histories[chain_idx] = llk_history


# =====================================================================
# FIGURE 1 : LOG-VRAISEMBLANCE
# =====================================================================
fig_llk, axes_llk = plt.subplots(num_chains, 1, figsize=(12, 2.5 * num_chains), sharex=False)
if num_chains == 1: axes_llk = [axes_llk]

for idx, (chain_idx, llk_hist) in enumerate(all_llk_histories.items()):
    ax = axes_llk[idx]
    ax.plot(np.arange(len(llk_hist)), llk_hist, label=f"Chaîne {chain_idx}", 
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
        if true_params_exist:
            ax.plot(x_faults, df_true_params.iloc[:, col_idx+1], label="Init/Cible", color="black", linestyle="None", marker="x", markersize=8)
        ax.plot(x_faults, mod_final[:, col_idx], label="Dernier PT", color="blue", linestyle="None", marker="o", alpha=0.8)
        
        ax.grid(True, linestyle=":", alpha=0.5)
        if row_idx == 0: ax.set_title(param_names[col_idx], fontweight='bold')
        if col_idx == 0: ax.set_ylabel(f"Chaîne {chain_idx}")
        if row_idx == 0 and col_idx == 0: ax.legend(fontsize=8)

fig_params.tight_layout()

# =====================================================================
# FIGURE 3 : DÉPLACEMENTS (3 SOUS-GRAPHIQUES SÉPARÉS)
# =====================================================================
try:
    df_data = pd.read_csv("surface_responses.csv")
    df_pred = pd.read_csv("last_model_responses.csv")

    fig_disp, axes_disp = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    components = ["North", "East", "Depth"]
    titles = ["Nord", "Est", "Profondeur (Vertical)"]
    
    for comp_idx, comp in enumerate(components):
        ax = axes_disp[comp_idx]
        col_name = f"St1_{comp}" 
        
        ax.plot(df_data["Time"], df_data[col_name], label="Cible (Data)", color="black", linestyle="--", linewidth=2, alpha=0.7)
        ax.plot(df_pred["Time"], df_pred[col_name], label="Prédiction PT", color="red", linewidth=1.5)
        
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if comp_idx == 0: ax.set_title(f"STATION 1 - {titles[comp_idx]}", fontweight='bold')
        else: ax.set_title(f"{titles[comp_idx]}")
        
        if comp_idx == 1: ax.set_ylabel("Déplacement (cm)")
        if comp_idx == 2: ax.set_xlabel("Temps (h)")
        if comp_idx == 0: ax.legend()

    fig_disp.tight_layout()
except FileNotFoundError:
    print("\nErreur : Fichiers CSV introuvables (surface_responses ou last_model_responses).")

# =====================================================================
# FIGURE 4 : TRACE MCMC (ÉVOLUTION DES PARAMÈTRES)
# =====================================================================
fig_trace, axes_trace = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for idx, (chain_idx, history) in enumerate(param_history.items()):
    steps = np.arange(len(history['k_a_sigma']))
    for i, p_key in enumerate(param_names):
        ax = axes_trace[i]
        ax.plot(steps, history[p_key], label=f"Chaîne {chain_idx}", color=colors[idx], alpha=0.7, linewidth=0.5)

for i, p_key in enumerate(param_names):
    ax = axes_trace[i]
    ax.set_ylabel(p_key, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if i == 0:
        ax.set_title("Trace MCMC : Historique d'exploration des 4 paramètres", fontweight='bold')
        ax.legend(loc="upper right")
    if i == 3:
        ax.set_xlabel("Pas MCMC (Transitions acceptées)")

fig_trace.tight_layout()

plt.show()