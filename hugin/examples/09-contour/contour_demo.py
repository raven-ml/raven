import numpy as np
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create grid
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
xx, yy = np.meshgrid(x, y)

# Create a simple 2D function: z = sin(sqrt(x^2 + y^2))
r = np.sqrt(xx**2 + yy**2)
z = np.sin(r)

# Create contour levels
levels = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]

# Add filled contour
contourf = ax.contourf(xx, yy, z, levels=levels, cmap='viridis')

# Add contour lines
contour = ax.contour(xx, yy, z, levels=levels, colors='black', linewidths=1.0)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Contour Plot Demo: sin(sqrt(x² + y²))')

# Save figure
print("Saving contour plot to contour_demo_python.png")
plt.savefig('contour_demo_python.png', dpi=100)
print("Done!")