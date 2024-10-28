import matplotlib.pyplot as plt

# Data
train_sizes = [150, 120, 90, 60, 30, 20, 15, 10, 5]
validation_losses = [12.88, 12.95, 13.01, 12.89, 13.05, 12.80, 12.64, 13.05, 12.95]
testing_losses = [12.81, 12.93, 12.93, 12.79, 12.92, 12.70, 13.19, 13.20, 12.87]
best_epochs = [6600, 6600, 7000, 7800, 6800, 7200, 6400, 7400, 7800]

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot training and testing losses on the first y-axis
ax1.set_xlabel('Training Size')
ax1.set_ylabel('Losses', color='tab:blue')
line1, = ax1.plot(train_sizes, validation_losses, 'o-', color='tab:orange', label='Validation Loss')
line2, = ax1.plot(train_sizes, testing_losses, 's-', color='tab:blue', label='Testing Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for best epochs
ax2 = ax1.twinx()
ax2.set_ylabel('Best Epoch', color='tab:gray')
line3, = ax2.plot(train_sizes, best_epochs, 'd--', color='tab:gray', label='Best Epoch')
ax2.tick_params(axis='y', labelcolor='tab:gray')

# Combine legends
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

# Title and show plot
plt.title('Losses and Best Epochs vs Training Size')
plt.savefig('train_and_test_losses_epochs_combined_legend.png', bbox_inches='tight')
plt.grid(True)
plt.show()
