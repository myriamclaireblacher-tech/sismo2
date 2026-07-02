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

# 1. Charger les données
file_path = 'surface_responses.csv'
df = pd.read_csv(file_path)

# 2. Configurer la figure
# On suppose qu'il y a N stations. On crée une ligne par station.
# Chaque ligne aura 3 colonnes pour North, East, Depth.
n_stations = (df.shape[1] - 1) // 3
fig, axes = plt.subplots(n_stations, 1, figsize=(10, 3 * n_stations), sharex=True)

# Gérer le cas où il n'y a qu'une seule station (axes n'est pas une liste)
if n_stations == 1:
    axes = [axes]

# 3. Tracer les données
for i in range(n_stations):
    # Les colonnes pour la station i+1
    col_n = f'St{i+1}_North'
    col_e = f'St{i+1}_East'
    col_d = f'St{i+1}_Depth'
    
    axes[i].plot(df['Time'], df[col_n], label='North')
    axes[i].plot(df['Time'], df[col_e], label='East')
    axes[i].plot(df['Time'], df[col_d], label='Depth')
    
    axes[i].set_title(f'Station {i+1}')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()
