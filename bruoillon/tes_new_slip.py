import matplotlib
matplotlib.use('TkAgg') 

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd


t_list=np.linspace( 0, 50, 300)

dtau = 2.0
V0=0.08
asigma=0.40
k=1

d_list = asigma/k * np.log(1+ np.exp(dtau/asigma)*1*(np.exp(k*V0*t_list/asigma)-1))


plt.plot(t_list,d_list)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger les deux jeux de données
df_obs = pd.read_csv('surface_responses.csv')
df_pred = pd.read_csv('best_results.csv')

# 2. Configurer la figure
n_stations = (df_obs.shape[1] - 1) // 3
fig, axes = plt.subplots(n_stations, 1, figsize=(11, 3 * n_stations), sharex=True)

if n_stations == 1:
    axes = [axes]

# 3. Tracer et superposer les données
for i in range(n_stations):
    col_n = f'St{i+1}_North'
    col_e = f'St{i+1}_East'
    col_d = f'St{i+1}_Depth'
    
    # --- DONNÉES OBSERVÉES (Lignes pleines) ---
    axes[i].plot(df_obs['Time'], df_obs[col_n], color='crimson', label='Obs North')
    axes[i].plot(df_obs['Time'], df_obs[col_e], color='royalblue', label='Obs East')
    axes[i].plot(df_obs['Time'], df_obs[col_d], color='forestgreen', label='Obs Depth')
    
    # --- MODÈLE INVERSÉ / PRÉDIT (Lignes pointillées -- ) ---
    axes[i].plot(df_pred['Time'], df_pred[col_n], color='darkred', linestyle='--', alpha=0.8, label='MCMC North')
    axes[i].plot(df_pred['Time'], df_pred[col_e], color='darkblue', linestyle='--', alpha=0.8, label='MCMC East')
    axes[i].plot(df_pred['Time'], df_pred[col_d], color='darkgreen', linestyle='--', alpha=0.8, label='MCMC Depth')
    
    axes[i].set_title(f'Station {i+1}')
    # Placement de la légende à l'extérieur ou en haut à droite
    axes[i].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize='small')
    axes[i].grid(True, linestyle=':', alpha=0.6)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()