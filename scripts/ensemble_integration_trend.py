

import os

from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt



def lorenz_NA(t, y): #non-autonomous case
    
    tau = 73 #one year is 73 units as one unit is 5 days
    #slope = 0
    slope = -2./(tau*100)
    F0 = 7
    if t>10*73:
        F0 = 7 + slope*(t-10*73)
    
    #F = (F0 + 2.0*np.cos(2*np.pi*t/tau))
    F = F0 + 2 * np.cos(2 * np.pi * t / tau) # seasonal forcing
    dy = [-y[1]**2 - y[2]**2 - a * y[0] + a * F,
          y[0] * y[1] - b * y[0] * y[2] - y[1] + G,
          b * y[0] * y[1] + y[0] * y[2] - y[2]]
    return dy

a = 0.25
b = 4.0
G = 1.0
F0 = 7.0
dt = 0.025

num_trajectories_pb = 10
pullback_initial_conditions = np.random.uniform(-3, 3, size=(num_trajectories_pb, 3))

years = 200
T = 73 * years
num_time_pts = int(T / dt)

#pullback trajectories for year 110
t = np.linspace(0, T, num_time_pts)
pullback_trajectories = []

for i in trange(num_trajectories_pb):
    initial_conditions = pullback_initial_conditions[i, :]
    solution = solve_ivp(lorenz_NA, (0, T), initial_conditions, t_eval=t)
    pullback_trajectories.append(solution.y)
    
pullback_trajectories = np.array(pullback_trajectories)

np.save('pullback_trajectories.npy', pullback_trajectories)

#########

#to select and save all the months of january or july in one array
for i in range(0, years):
    january = pullback_trajectories[:, :, int((0+73*i)/dt):int((0+73*i)/dt)+240]
    july = pullback_trajectories[:, :, int((36+73*i)/dt):int((36+73*i)/dt)+240]
    np.save('january_' + str(i) + '.npy', january)
    np.save('july_' + str(i) + '.npy', july)
    
    

# Define the directory where the npy files are located

script_directory = os.path.dirname(os.path.abspath(__file__))
npy_directory = script_directory
# Create an empty list to store the NetCDF file paths
npy_files = []

# Iterate through the files in the directory and add NetCDF files to the list
for filename in os.listdir(npy_directory):
    if filename.endswith(".npy"):
        npy_files.append(os.path.join(npy_directory, filename))

# Initialize an empty list to store data from all files
pullback_trajectories_list = []

# Loop through each npy file and read data
for npy_file in npy_files:
    pullback_trajectories = np.load(npy_file)
    pullback_trajectories_list.append(pullback_trajectories)



# Concatenate the data from all files
concatenated_pullback_trajectories = np.concatenate(pullback_trajectories_list)



