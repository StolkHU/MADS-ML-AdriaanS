import pandas as pd
import matplotlib.pyplot as plt

# Create the DataFrame
data = {
    "Experiment": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ],
    "Hidden_Layers": [
        256,
        258,
        300,
        300,
        256,
        256,
        300,
        300,
        258,
        200,
        200,
        200,
        256,
        256,
        512,
        128,
        100,
        100,
        256,
        300,
    ],
    "Epochs": [
        100,
        200,
        100,
        100,
        100,
        200,
        100,
        100,
        100,
        200,
        200,
        200,
        200,
        100,
        100,
        100,
        100,
        100,
        200,
        100,
    ],
    "GRU": [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ],
    "GRU_Layers": [
        2,
        2,
        2,
        2,
        0.75,
        1,
        2,
        3,
        0.05,
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        2,
    ],
    "Convolutional_Layer": [
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ],
    "LR_Factor": [
        0.1,
        0.11,
        0.09,
        0.15,
        0.75,
        0.1,
        0.2,
        0.1,
        0.05,
        0.031,
        0.33,
        0.33,
        0.75,
        0.25,
        0.05,
        None,
        None,
        None,
        0.1,
        0.2,
    ],
    "Epochs_Trained": [
        10,
        10,
        10,
        8,
        10,
        10,
        10,
        10,
        5,
        None,
        20,
        20,
        10,
        10,
        5,
        None,
        None,
        None,
        10,
        10,
    ],
    "Loss_Test": [
        0.0172,
        0.0283,
        0.0307,
        0.0364,
        0.033,
        0.0338,
        0.0325,
        0.0289,
        0.0372,
        0.031,
        0.0408,
        0.053,
        0.0501,
        0.0568,
        0.0619,
        0.0516,
        0.0968,
        0.1025,
        0.074,
        0.1334,
    ],
    "Loss_Train": [
        0.0002,
        0.0003,
        0.0003,
        0.0003,
        0.0009,
        0.0003,
        0.0003,
        0.0002,
        0.0021,
        0.0013,
        0.0014,
        0.0011,
        0.0001,
        0.0004,
        0.002,
        0.0038,
        0.0065,
        0.0109,
        0.0008,
        0.1761,
    ],
    "Accuracy": [
        0.9953,
        0.9938,
        0.9938,
        0.9938,
        0.9938,
        0.9922,
        0.9922,
        0.9922,
        0.9922,
        0.9922,
        0.9906,
        0.9906,
        0.9891,
        0.9844,
        0.9844,
        0.9844,
        0.9844,
        0.9781,
        0.9766,
        0.9688,
    ],
}

df = pd.DataFrame(data)

# Visualization 1: Loss Comparison
plt.figure(figsize=(12, 6))
plt.scatter(df["Loss_Test"], df["Loss_Train"], c=df["Accuracy"], cmap="viridis")
plt.colorbar(label="Accuracy")
plt.xlabel("Test Loss")
plt.ylabel("Train Loss")
plt.title("Test Loss vs Train Loss (Colored by Accuracy)")
plt.tight_layout()
plt.savefig("loss_comparison.png")
plt.close()

# Visualization 2: Hyperparameter Impact on Accuracy
plt.figure(figsize=(12, 6))
plt.scatter(df["Hidden_Layers"], df["Accuracy"], c=df["Loss_Test"], cmap="coolwarm")
plt.colorbar(label="Test Loss")
plt.xlabel("Hidden Layers")
plt.ylabel("Accuracy")
plt.title("Hidden Layers vs Accuracy (Colored by Test Loss)")
plt.tight_layout()
plt.savefig("hidden_layers_accuracy.png")
plt.close()

# Statistical Summary
print("Statistical Summary:")
print(df[["Hidden_Layers", "Epochs", "Loss_Test", "Loss_Train", "Accuracy"]].describe())

# Correlation Analysis
print("\nCorrelation Matrix:")
correlation_matrix = df[
    ["Hidden_Layers", "Epochs", "Loss_Test", "Loss_Train", "Accuracy"]
].corr()
print(correlation_matrix)

# Find Best Performing Experiments
print("\nTop 5 Experiments by Accuracy:")
print(
    df.nlargest(5, "Accuracy")[
        ["Experiment", "Hidden_Layers", "Epochs", "Loss_Test", "Loss_Train", "Accuracy"]
    ]
)

print("\nTop 5 Experiments by Lowest Test Loss:")
print(
    df.nsmallest(5, "Loss_Test")[
        ["Experiment", "Hidden_Layers", "Epochs", "Loss_Test", "Loss_Train", "Accuracy"]
    ]
)
