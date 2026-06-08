#écrit par une ia j'avais trop la flemme
#modify here
#terminal must be in the file of file_name

import pandas as pd
import matplotlib.pyplot as plt

"""
file_name='test.csv'




try:
    data = pd.read_csv(file_name)
except FileNotFoundError:
    print("Error: The file 'test.csv' was not found.")
    exit()

print("Columns read:", list(data.columns))

t = data.iloc[:, 0]
slip = data.iloc[:, 1]
V = data.iloc[:, 2]

plt.figure(figsize=(10, 5))
plt.plot(t, slip, color='blue', linewidth=2)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Slip (cm)', fontsize=12)
plt.title('Evolution of slip over time', fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t, V, color='black', linewidth=2)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Slip velocity (cm/hour)', fontsize=12)
plt.title('Evolution of slip velocity over time', fontsize=14)
plt.grid(True)
plt.show()
"""

# Chargement des données du fichier CSV
df = pd.read_csv("surface_responses.csv")

# Configuration de la figure (3 sous-graphiques : 1 par composante)
fig, (ax_north, ax_east, ax_depth) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Définition des couleurs : 1 par station
color_st1 = "blue"
color_st2 = "orange"

# 1. Sous-graphique NORD
ax_north.plot(df["Time"], df["St1_North"], label="Station 1", color=color_st1, linewidth=1.5)
ax_north.plot(df["Time"], df["St2_North"], label="Station 2", color=color_st2, linewidth=1.5)
ax_north.set_ylabel("Déplacement (cm)")
ax_north.set_title("Composante Nord")
ax_north.grid(True, linestyle=":", alpha=0.6)
ax_north.legend()

# 2. Sous-graphique EST
ax_east.plot(df["Time"], df["St1_East"], label="Station 1", color=color_st1, linewidth=1.5)
ax_east.plot(df["Time"], df["St2_East"], label="Station 2", color=color_st2, linewidth=1.5)
ax_east.set_ylabel("Déplacement (cm)")
ax_east.set_title("Composante Est")
ax_east.grid(True, linestyle=":", alpha=0.6)
ax_east.legend()

# 3. Sous-graphique PROFONDEUR / VERTICAL
ax_depth.plot(df["Time"], df["St1_Depth"], label="Station 1", color=color_st1, linewidth=1.5)
ax_depth.plot(df["Time"], df["St2_Depth"], label="Station 2", color=color_st2, linewidth=1.5)
ax_depth.set_xlabel("Temps (h)")
ax_depth.set_ylabel("Déplacement (cm)")
ax_depth.set_title("Composante Profondeur (Vertical)")
ax_depth.grid(True, linestyle=":", alpha=0.6)
ax_depth.legend()

plt.tight_layout()
plt.show()