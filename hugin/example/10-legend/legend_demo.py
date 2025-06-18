import numpy as np
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create some data
x = np.linspace(0.0, 10.0, 100)

# Plot multiple functions with labels
ax.plot(x, np.sin(x), color='blue', label='sin(x)')
ax.plot(x, np.cos(x), color='red', label='cos(x)')
ax.plot(x, np.sin(x) + np.cos(x), color='green', linestyle='--', label='sin(x) + cos(x)')

# Add scatter plot
x_scatter = np.linspace(0.0, 10.0, 20)
y_scatter = np.sin(x_scatter) * 0.8
ax.scatter(x_scatter, y_scatter, color='orange', marker='o', s=30, label='data points')

# Enable legend
ax.legend()

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Legend Demo: Trigonometric Functions')

# Enable grid
ax.grid(True)

# Save figure
print("Saving legend demo to legend_demo_python.png")
plt.savefig('legend_demo_python.png', dpi=100)
print("Done!")