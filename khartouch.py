import matplotlib.pyplot as plt

# Data
train_sizes = [1000, 464, 215, 100, 46, 22, 10, 5, 2, 1]
validation_losses = [12.69, 12.92, 12.86, 13.03, 13.03, 13.02, 12.99, 13.19, 12.87, 13.03]
testing_losses = [12.71, 12.93, 12.93, 13.06, 13.00, 13.03, 12.87, 13.61, 13.13, 13.39]


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, validation_losses, label='Validation Loss', marker='o', color='blue')
plt.plot(train_sizes, testing_losses, label='Testing Loss', marker='s', color='red')

plt.xscale('log')

# Adding labels and title
plt.xlabel('Training Sizes')
plt.ylabel('Loss')
plt.title('Validation and Testing Losses vs. Training Sizes')
plt.xscale('log')  # Optional: log scale to better visualize the spread
plt.legend()
plt.grid(True, which="both")

plt.savefig("validation and testing losses vs training size .png")

# Show plot
plt.show()