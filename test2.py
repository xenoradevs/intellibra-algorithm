import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ... Your code for simulating data ...

# Sensor Positions (Approximate Arc)
num_sensors_per_set = 3
theta_set_1 = np.linspace(0, np.pi, num_sensors_per_set, endpoint=False) 
theta_set_2 = np.linspace(np.pi, 2 * np.pi, num_sensors_per_set, endpoint=False)
theta = np.concatenate((theta_set_1, theta_set_2))

radius = 0.005  
phi = np.full_like(theta, 0.5 * np.pi) 

def spherical_to_xyz(theta, phi, radius):
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert spherical coordinates to Cartesian
x, y, z = spherical_to_xyz(theta, phi, radius)

ax.scatter(x, y, z, s=1000, c='blue')  
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Approximate Sensor Positions')
plt.show()
