import matplotlib.pyplot as plt
import pandas as pd

# Je data
data = {"Type": ["CNN", "Transfer Learning"], "Accuracy": [58.1105, 93.1590]}

# Maak DataFrame
df = pd.DataFrame(data)

# Maak een barplot
plt.figure(figsize=(8, 6))
bars = plt.bar(df["Type"], df["Accuracy"], color=["silver", "green"], alpha=0.7)

plt.ylabel("Gemiddelde Accuracy (%)")
plt.title("Gemiddelde Model Performance Vergelijking")

# Voeg percentages boven de bars toe
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Zet Y-as van 0 tot 100
plt.ylim(0, 100)

# Sla op als PNG
plt.savefig("model_performance_barplot.png", dpi=300, bbox_inches="tight")
plt.show()
