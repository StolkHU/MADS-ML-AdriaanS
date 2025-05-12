import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the data
data = {
    "Batch_Norm": ["No", "No", "No", "No", "Yes", "No", "Yes"],
    "Dropout": ["No", "No", "No", "No", "No", "Yes", "Yes"],
    "Filters": [16, 72, 128, 72, 72, 72, 72],
    "Loss_test": [0.2901, 0.2751, 0.32, 0.2751, 0.2918, 0.2546, 0.2334],
    "Loss_train": [0.25, 0.1133, 0.0739, 0.1133, 0.0769, 0.2812, 0.2543],
    "Accuracy": [0.8965, 0.9187, 0.9134, 0.9187, 0.9118, 0.9081, 0.9175],
}

df = pd.DataFrame(data)

# Create configuration labels
df["Config"] = df.apply(
    lambda row: f"BN:{row['Batch_Norm'][0]}/DO:{row['Dropout'][0]}/F:{row['Filters']}",
    axis=1,
)

# Create the overfitting analysis plot
plt.figure(figsize=(10, 8))

# Create scatter plot of test vs train loss
scatter = plt.scatter(
    df["Loss_train"],
    df["Loss_test"],
    s=df["Filters"] * 2,
    alpha=0.7,
    c=df["Accuracy"],
    cmap="viridis",
    edgecolors="black",
    linewidth=2,
)

# Add diagonal line for reference (perfect generalization)
max_loss = max(df["Loss_train"].max(), df["Loss_test"].max())
plt.plot(
    [0, max_loss],
    [0, max_loss],
    "r--",
    alpha=0.5,
    linewidth=2,
    label="Perfecte generalisatie",
)

# Annotate points
for idx, row in df.iterrows():
    plt.annotate(
        row["Config"],
        (row["Loss_train"], row["Loss_test"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

plt.xlabel("Training Loss", fontsize=12, fontweight="bold")
plt.ylabel("Test Loss", fontsize=12, fontweight="bold")
plt.title(
    "Dropout zorgt niet voor overfitting\n(Boven de lijn = Overfitting)",
    fontsize=14,
    fontweight="bold",
)
plt.legend()
plt.grid(True, alpha=0.3)

# Add colorbar for accuracy
cbar = plt.colorbar(scatter)
cbar.set_label("Model Accuracy", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("overfitting_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# Print overfitting ratios
print("\nOverfitting Ratios (Test Loss / Train Loss):")
print("=" * 40)
df["Overfitting_Ratio"] = df["Loss_test"] / df["Loss_train"]
for idx, row in df.iterrows():
    print(f"{row['Config']:20} : {row['Overfitting_Ratio']:.3f}")
print(f"\nLower ratio = Better generalization")
print(
    f"Best generalization: {df.loc[df['Overfitting_Ratio'].idxmin(), 'Config']} (ratio: {df['Overfitting_Ratio'].min():.3f})"
)
