import numpy as np
import matplotlib.pyplot as plt

# Create sample data
def create_data():
    # Create x values from 0 to 2π with 100 points
    x = np.linspace(0, 2 * np.pi, 100)
    # Calculate y = sin(x)
    y = np.sin(x)
    return x, y

# Generate the data
x, y = create_data()

# Create the plot
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")

# Display the plot
plt.show()