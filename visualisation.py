
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

# Configuration de la figure (2 sous-graphiques pour séparer les stations)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Tracé pour la Station 1
ax1.plot(df["Time"], df["St1_North"], label="Nord", color="darkred")
ax1.plot(df["Time"], df["St1_East"], label="Est", color="darkgreen")
ax1.plot(df["Time"], df["St1_Depth"], label="Profondeur", color="darkblue")
ax1.set_ylabel("Déplacement (cm)")
ax1.set_title("Station 1")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

# Tracé pour la Station 2
ax2.plot(df["Time"], df["St2_North"], label="Nord", color="crimson", linestyle="--")
ax2.plot(df["Time"], df["St2_East"], label="Est", color="forestgreen", linestyle="--")
ax2.plot(df["Time"], df["St2_Depth"], label="Profondeur", color="royalblue", linestyle="--")
ax2.set_xlabel("Temps (h)")
ax2.set_ylabel("Déplacement (cm)")
ax2.set_title("Station 2")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()