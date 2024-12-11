#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import trange  # Import trange from tqdm for progress bar

def lorenz(y):
    dy = [-y[1]**2 - y[2]**2 - a*y[0] + a*F, y[0]*y[1] - b*y[0]*y[2] - y[1] + G, b*y[0]*y[1] + y[0]*y[2] - y[2]]
    return dy

# Set the parameters
a = 0.25
b = 4.0
G = 1.0
#F0 = 7.0
dt = 0.025
years = 6
T = 73 * years
num_time_pts = int(T / dt)
t = np.linspace(0, T, num_time_pts)

# Range of F values. here we're looking at the inset around the Hopf bifurcation
F_range = np.arange(1.1, 1.4, 0.002)

# Create a figure
fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
ax.set_xlabel('F', fontsize=24)
ax.set_ylabel('X', fontsize=24, rotation=0)
#set ticks size
ax.tick_params(axis='both', which='major', labelsize=20)
#ax.set_title('Last 10 points of X vs F for 10 different runs')

# Iterate over the range of F values with a progress bar
for i in trange(len(F_range), desc="Processing F values"):
    F = F_range[i]
    
    for j in range(100):  # Perform 10 different runs
        # Generate random initial conditions
        y0 = np.random.rand(3) * 2 - 1  # Random values in range [-1, 1]
        
        # Solve the ODE
        sol = solve_ivp(lorenz, [0, T], y0, t_eval=t, method="RK45")
        
        # Extract the last 10 points
        y = sol.y[:, -100:]
        
        # Plot the results
        ax.plot(F * np.ones(100), y[0, :], 'b.', markersize=.05)

# Draw vertical lines and fill areas with transparency
ax.set_ylim(-1, 1.5)

# Add a legend and save the figure
fig.legend()
# %%
ax.set_ylim(-1, 1.5)
#set ax label size font
ax.set_xlabel('F', fontsize=15)
ax.set_ylabel('X', fontsize=15, rotation=0)
#set ticks size
ax.tick_params(axis='both', which='major', labelsize=12)
fig
# %%
