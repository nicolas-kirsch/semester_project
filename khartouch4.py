import matplotlib.pyplot as plt

# Data for plotting
data = [
    {"Dim-internal": 10, "l": 8, "validation_loss": 13.19, "test_loss": 13.61, "delta": 0.42},
    {"Dim-internal": 6, "l": 8, "validation_loss": 13.10, "test_loss": 13.04, "delta": -0.06},
    {"Dim-internal": 4, "l": 8, "validation_loss": 12.70, "test_loss": 12.73, "delta": 0.03},
    {"Dim-internal": 10, "l": 5, "validation_loss": 13.00, "test_loss": 13.29, "delta": 0.29},
    {"Dim-internal": 6, "l": 5, "validation_loss": 12.88, "test_loss": 13.10, "delta": 0.22},
    {"Dim-internal": 4, "l": 5, "validation_loss": 12.82, "test_loss": 13.12, "delta": 0.3},
    {"Dim-internal": 10, "l": 3, "validation_loss": 12.69, "test_loss": 13.17, "delta": 0.48},
    {"Dim-internal": 6, "l": 3, "validation_loss": 13.27, "test_loss": 13.45, "delta": 0.18},
    {"Dim-internal": 4, "l": 3, "validation_loss": 12.96, "test_loss": 13.42, "delta": 0.46},
]

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot data points
for entry in data:
    dim = entry["Dim-internal"]
    l = entry["l"]
    val_loss = entry["validation_loss"]
    test_loss = entry["test_loss"]

    plt.scatter([dim], [val_loss], label=f'Validation loss, l={l}', marker='o', s=80, alpha=0.7)
    plt.scatter([dim], [test_loss], label=f'Test loss, l={l}', marker='x', s=80, alpha=0.7)

# Adding labels and title
plt.xlabel("Dim-internal")
plt.ylabel("Loss")
plt.title("Validation and Test Losses for Different Dim-internal and l Values")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Display plot
plt.show()
