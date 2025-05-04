import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Units
units = [256, 128, 64]

data = [
    [0.8792, 0.8748, 0.8610],  # 2 lagen
    [0.8751, 0.8729, 0.8763],  # 3 lagen
    [0.8707, 0.8776, 0.8716],  # 4 lagen
]

# Creëer een DataFrame
df = pd.DataFrame(data)
df.columns = units
df.index = [2, 3, 4]
df.index.name = "Layers"
df.columns.name = "Units"

# Bereken statistieken voor kleurkeuzes
mean_val = np.mean(data)
std_val = np.std(data)
high_threshold = mean_val + std_val
low_threshold = mean_val - std_val

colors = ["#d13636", "#f0f0f0", "#f9f9f9", "#e0e0e0", "#c0c0c0", "#36a64d"]
positions = [0, 0.3, 0.45, 0.55, 0.7, 1]
cmap = LinearSegmentedColormap.from_list(
    "custom_gray_highlight", list(zip(positions, colors))
)

# Maak een heatmap
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(
    df, annot=True, cmap=cmap, fmt=".4f", linewidths=0.5, cbar_kws={"label": "Accuracy"}
)

plt.title("Heatmap van Model Accuracy", fontsize=16, fontweight="bold")
plt.xlabel("Units", fontsize=14)
plt.ylabel("Layers", fontsize=14)

# Verbeter leesbaarheid
plt.tight_layout()

# Toon de afbeelding
plt.savefig("Accuracy Week 1.png", dpi=300, bbox_inches="tight")
plt.show()
