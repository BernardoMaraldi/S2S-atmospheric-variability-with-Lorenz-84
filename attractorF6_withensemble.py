#%%
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt



def lorenz_A(t, y): #non-autonomous case
    
    #tau = 73 #one year is 73 units as one unit is 5 days    
    F0 = 6 
    F = F0 
    dy = [-y[1]**2 - y[2]**2 - a * y[0] + a * F,
          y[0] * y[1] - b * y[0] * y[2] - y[1] + G,
          b * y[0] * y[1] + y[0] * y[2] - y[2]]
    return dy

a = 0.25
b = 4.0
G = 1.0
F = 6
dt = 0.025

num_trajectories_fw = 50000
forward_initial_conditions = np.random.uniform(-3, 3, size=(num_trajectories_fw, 3))

years = 5 #make sure the system converges on the attractor
T = 73 * years
num_time_pts = int(T / dt)

#pullback trajectories for year 110
t = np.linspace(0, T, num_time_pts)
forward_trajectories = []

for i in trange(num_trajectories_fw):
    initial_conditions = forward_initial_conditions[i, :]
    solution = solve_ivp(lorenz_A, (0, T), initial_conditions, t_eval=t)
    forward_trajectories.append(solution.y)
    
forward_trajectories_F6 = np.array(forward_trajectories)
# %%
#store only the last 10 points of each trajectory
last_points_fw = forward_trajectories_F6[:, :, -1000:]
#distinguish in two groups: if the min of x value is zero assign to group 1, otherwise to group 2
group1 = []
group2 = []
for i in range(num_trajectories_fw):
    if np.max(last_points_fw[i, 1, :]) >= 1.5:
        group1.append(last_points_fw[i, :, :])
    else:
        group2.append(last_points_fw[i, :, :])
group1 = np.array(group1)
group2 = np.array(group2)

#%%
# plot the 2 groups in 3d in the same plot with different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(group1)):
    ax.plot(group1[i, 0, -50:], group1[i, 1, -50:], group1[i, 2, -50:], 'r', alpha=0.05)
for i in range(len(group2)):
    ax.plot(group2[i, 0, -50:], group2[i, 1, -50:], group2[i, 2, -50:], 'b', alpha=0.05)
plt.show()

#%%
#plot the last 10 points of each trajectory in 2d xy
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(forward_trajectories_F6)):
    ax.plot(forward_trajectories_F6[i, 2, :], forward_trajectories_F6[i, 1, :], 'r') 
# %%
#separate the trajectories in 
#integrate a trajectory starting at 2.4,1,0
initial_conditions = [2.4, 1, 0]
solution = solve_ivp(lorenz_A, (0, T), initial_conditions, t_eval=t)
plt.plot(solution.y[2, -1000:], solution.y[1, -1000:])

# %%
