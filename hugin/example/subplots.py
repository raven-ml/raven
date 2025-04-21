import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("Generating Matplotlib/NumPy demo plot...")

# --- Figure Setup ---
# Hugin uses pixels, Matplotlib uses inches. Adjust figsize accordingly.
# 1600x1200 pixels at 100 DPI = 16x12 inches.
fig = plt.figure(figsize=(16, 12))
nrows, ncols = 4, 3

# --- 1. Basic Plot (Line2D) ---
ax1 = fig.add_subplot(nrows, ncols, 1)
x1 = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x1)
y1_cos = np.cos(x1)

ax1.plot(x1, y1, color='blue', linestyle='-', marker='o', label='Sine')
ax1.plot(x1, y1_cos, color='red', linestyle='--', label='Cosine')
ax1.set_title("1. Basic Plot (plot)")
ax1.set_xlabel("Radians")
ax1.set_ylabel("Value")
ax1.grid(True)
ax1.text(np.pi / 2.0, 0.5, "Annotation", color='darkgray')
ax1.legend()

# --- 2. Plot Y only ---
ax2 = fig.add_subplot(nrows, ncols, 2)
x_for_y2 = np.linspace(0, 20, 100)
y2 = np.exp(-x_for_y2 / 10.0) * np.sin(x_for_y2)

ax2.plot(y2, color='green', marker='.', linestyle='None', label='exp(-x/10)*sin(x)') # Use plot with linestyle='None' for points
ax2.set_title("2. Plot Y-Data Only (plot_y)")
ax2.set_xlabel("Index")
ax2.set_ylabel("Value")
ax2.grid(True)
ax2.legend()

# --- 3. Scatter Plot ---
ax3 = fig.add_subplot(nrows, ncols, 3)
# Generate random data similar to OCaml example
x3 = np.random.rand(50) * 10.0
y3 = x3 + (np.random.rand(50) * 2.0 - 1.0) # Add noise [-1, 1]
scatter_color = (0.8, 0.2, 0.8, 0.6) # RGBA tuple

ax3.scatter(x3, y3, s=50.0, c=[scatter_color], marker='*', label='Noisy Data') # Note: c needs to be sequence
ax3.set_title("3. Scatter Plot")
ax3.set_xlabel("X Value")
ax3.set_ylabel("Y Value")
ax3.grid(True)
ax3.legend()

# --- 4. Bar Chart ---
ax4 = fig.add_subplot(nrows, ncols, 4)
x4 = np.arange(5) # Use integer indices for categories
height4 = (x4 + 1) * 1.5

ax4.bar(x4, height4, width=0.8, color='orange', label='Categories')
ax4.set_title("4. Bar Chart")
ax4.set_xlabel("Category Index")
ax4.set_ylabel("Height")
ax4.set_xticks(x4) # Set ticks at the center of the bars
ax4.legend()

# --- 5. Histogram ---
ax5 = fig.add_subplot(nrows, ncols, 5)
# Generate standard normal data
data5 = np.random.randn(1000)

ax5.hist(data5, bins=20, color='cyan', density=True, label='Normal Distribution')
ax5.set_title("5. Histogram")
ax5.set_xlabel("Value")
ax5.set_ylabel("Density")
ax5.grid(True, axis='y')
ax5.legend()

# --- 6. Step Plot ---
ax6 = fig.add_subplot(nrows, ncols, 6)
x6 = np.linspace(0, 10, 11)
y6 = np.sin(x6 / 2.0)
y6_shifted = y6 - 0.5

ax6.step(x6, y6, where='mid', color='magenta', linestyle='-.', label='Mid Step')
ax6.step(x6, y6_shifted, where='post', color='darkgray', label='Post Step')
ax6.set_title("6. Step Plot")
ax6.set_xlabel("X")
ax6.set_ylabel("Y")
ax6.grid(True)
ax6.legend()

# --- 7. Fill Between ---
ax7 = fig.add_subplot(nrows, ncols, 7)
x7 = np.linspace(0, 2 * np.pi, 100)
y7_base = np.sin(x7)
y7a = y7_base + 0.5
y7b = y7_base - 0.5
fill_color = (0.5, 0.8, 0.5) # RGB tuple
fill_alpha = 0.5

ax7.fill_between(x7, y7a, y7b, color=fill_color, alpha=fill_alpha, interpolate=True, label='Filled Sine Band')
ax7.plot(x7, y7a, color='black', linewidth=0.5) # Show boundary
ax7.plot(x7, y7b, color='black', linewidth=0.5) # Show boundary
ax7.set_title("7. Fill Between")
ax7.set_xlabel("X")
ax7.set_ylabel("Y")
ax7.legend()


# --- 8. Error Bars ---
ax8 = fig.add_subplot(nrows, ncols, 8)
x8 = np.linspace(0.5, 5.5, 6)
y8 = 1.0 / x8 + (np.random.rand(6) * 0.2 - 0.1) # Add noise [-0.1, 0.1]
yerr8 = np.random.rand(6) * 0.1 + 0.05
xerr8 = np.random.rand(6) * 0.2 + 0.1

# Hugin fmt combines style; Matplotlib separates fmt string or uses keywords
ax8.errorbar(x8, y8, yerr=yerr8, xerr=xerr8,
             marker='s', color='red', linestyle='None', # Style for central points
             ecolor='gray', elinewidth=1.0, capsize=4.0, # Error bar style
             label='Data +/- Error')
ax8.set_title("8. Error Bars")
ax8.set_xlabel("X")
ax8.set_ylabel("1/X + Noise")
ax8.grid(True)
ax8.legend()


# --- 9. Image Show (imshow) ---
ax9 = fig.add_subplot(nrows, ncols, 9)
rows9, cols9 = 10, 10
# Create data similar to OCaml example (ramp from 0 to 255)
data9 = (np.arange(rows9 * cols9).reshape(rows9, cols9) * 2.55)
data9 = np.clip(data9, 0, 255).astype(np.uint8)

# extent=(left, right, bottom, top)
im9 = ax9.imshow(data9, cmap='viridis', aspect='equal',
                 extent=(0, cols9, 0, rows9), origin='lower') # Match Hugin extent, use origin='lower' if extent bottom=0
ax9.set_title("9. Image Show (imshow)")
ax9.set_xlabel("Column Index")
ax9.set_ylabel("Row Index")
fig.colorbar(im9, ax=ax9) # Add colorbar


# --- 10. Matrix Show (matshow) ---
ax10 = fig.add_subplot(nrows, ncols, 10)
rows10, cols10 = 8, 8
# Use np.fromfunction for concise generation
def f10(r, c):
  return np.sin(r / 2.0) * np.cos(c / 3.0)
data10 = np.fromfunction(f10, (rows10, cols10))

# matshow sets origin='upper', aspect='equal' by default and styles axes
im10 = ax10.matshow(data10, cmap='coolwarm') # Origin 'upper' is default
ax10.set_title("10. Matrix Show (matshow)")
# matshow often disables axis labels, but we can add them if needed
# ax10.set_xlabel("Column")
# ax10.set_ylabel("Row")
ax10.xaxis.set_ticks_position('bottom') # Position ticks if shown
fig.colorbar(im10, ax=ax10)


# --- 11. 3D Plot (Line3D) ---
# Need explicit projection='3d'
ax11 = fig.add_subplot(nrows, ncols, 11, projection='3d')
t11 = np.linspace(0, 6 * np.pi, 200)
scale = (1.0 + t11 / (6 * np.pi))
x11 = np.cos(t11) * scale
y11 = np.sin(t11) * scale
z11 = t11 / (2 * np.pi)

ax11.plot(x11, y11, z11, color='blue', linewidth=2.0, label='Spiral')
ax11.set_title("11. 3D Plot (plot3d)")
ax11.set_xlabel("X")
ax11.set_ylabel("Y")
ax11.set_zlabel("Z")
ax11.view_init(elev=30, azim=45) # Set view angle
ax11.legend()


# --- 12. Placeholder / Combined Example ---
ax12 = fig.add_subplot(nrows, ncols, 12)
x12 = np.linspace(-5, 5, 50)
y12_tanh = np.tanh(x12)
# Use np.where for step function logic
y12_step = np.where(x12 > 0, 1.0, -1.0)

ax12.plot(x12, y12_tanh, color='black', label='tanh(x)')
ax12.step(x12, y12_step, color='red', where='mid', label='step(x)')
ax12.set_title("12. Combined Plot/Step")
ax12.set_xlabel("X")
ax12.set_ylabel("Y")
ax12.set_ylim(-1.5, 1.5)
ax12.grid(True)
ax12.legend()

plt.tight_layout()

print("Showing plot window...")
plt.show()

print("Done.")