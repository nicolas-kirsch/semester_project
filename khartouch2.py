import numpy as np
import matplotlib.pyplot as plt

# Original and flat heat demand profiles
original_heat_demand =  [30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 60, 80, 100, 90, 80, 70, 50, 40]
# original_heat_demand += original_heat_demand
flat_heat_demand =  [30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 55, 55, 55, 55, 55, 55, 55, 55]
flat_heat_demand += flat_heat_demand
flat_heat_demand =  [30, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

# Smoothing window size
window_size = 3

# Apply convolution without padding for the original_heat_demand
original_smoothed = -np.convolve(original_heat_demand, np.ones(window_size) / window_size, mode='same')

# Pad the flat_heat_demand array by replicating the first and last values
pad_width = window_size // 2
flat_padded = np.pad(flat_heat_demand, pad_width, mode='edge')

# Apply convolution with padding for the flat_heat_demand
flat_smoothed = -np.convolve(flat_padded, np.ones(window_size) / window_size, mode='valid')

# Time or index for x-axis
time = np.arange(len(original_heat_demand))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time, original_smoothed, label='Original Heat Demand', linestyle='-', marker='o')
plt.plot(time, flat_smoothed, label='Drop Flat Heat Demand', linestyle='--', marker='s')

# Adding labels and title
plt.xlabel('Time (Hours)')
plt.ylabel('Heat Demand (Smoothed)')
plt.title('Comparison of Heat Demand Profiles (Original vs Drop Flat)')
plt.legend()
plt.grid(True)

plt.savefig('Comparison of original vs drop flat demand profiles')
# Show the plot
plt.show()
