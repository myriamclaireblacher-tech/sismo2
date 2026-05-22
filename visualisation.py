#modify here
#terminal must be in the file of file_name
file_name='test.csv'


import pandas as pd
import matplotlib.pyplot as plt

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