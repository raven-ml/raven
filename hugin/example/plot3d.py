import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_helix_data():
    # Create 100 points from 0 to 4π
    t = np.linspace(0, 4 * np.pi, 100)
    # x = cos(t), y = sin(t), z = t / (4π)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi)  # Linear z scaled from 0 to 1
    return x, y, z

def main():
    # Generate data
    x, y, z = create_helix_data()
    print("x:", x)
    print("\n")
    print("y:", y)
    print("\n")
    print("z:", z)
    print("\n")
    
    # Create a figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D helix
    ax.plot(x, y, z, color='b', linewidth=1.0)
    
    # Set labels and title
    ax.set_title("3D Helix")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
