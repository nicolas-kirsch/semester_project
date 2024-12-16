# this is to plot the original heat demand only

import numpy as np
import matplotlib.pyplot as plt

# Original and flat heat demand profiles
original_heat_demand =  [30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 60, 80, 100, 90, 80, 70, 50, 40]

# Smoothing window size
window_size = 3

# Apply convolution without padding for the original_heat_demand
original_smoothed = -np.convolve(original_heat_demand, np.ones(window_size) / window_size, mode='same')

# Time or index for x-axis
time = np.arange(len(original_heat_demand))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time, original_smoothed, label='Original Heat Demand', linestyle='-', marker='o')

# Adding labels and title
plt.xlabel('Time (Hours)')
plt.ylabel('°C')
plt.title('Heat demand profile')
plt.grid(True)

plt.savefig('head demand profile .png')
# Show the plot
plt.show()
