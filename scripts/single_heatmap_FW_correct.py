#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #I 
import copy
import matplotlib.colors as colors
from tqdm import trange

my_cmap = copy.copy(plt.cm.get_cmap('hot')) # copy the default cmap
my_cmap.set_bad((0,0,0))

def heatmaps_plot(ax, anno, xx, yy, year):
    heatmap_p = np.histogram2d(
        anno[xx, :],
        anno[yy, :],
        bins=(500, 500),
        #range=[[-2.5, 2.5], [-2.5, 2.5]]
        range=[[-1, 1], [-1, 1]]
    )[0]
    
    #extent = [-2.5, 2.5, -2.5, 2.5]
    extent = [-1, 1, -1, 1]


    vmin = np.min(heatmap_p[heatmap_p > 0])  # Exclude zeros for LogNorm
    vmax = np.max(heatmap_p)
    
    ax.imshow(heatmap_p.T, extent=extent, origin='lower', aspect='auto', 
             norm=colors.LogNorm(vmin=vmin, vmax=vmax),
           cmap=my_cmap)# vmin=0)
    
    # ax.imshow(heatmap_p.T, extent=extent, origin='lower', aspect='auto',
    #        cmap='pink', vmin=0)
    
    if xx == 0:
        ax.set_xlabel('X', rotation=0, fontsize=25)
    elif xx == 1:
        ax.set_xlabel('Y', rotation=0, fontsize=25)
    elif xx == 2:
        ax.set_xlabel('Z', rotation=0, fontsize=25)
    
    if yy == 0:
        ax.set_ylabel('X', rotation=0, fontsize=25)
    elif yy == 1:
        ax.set_ylabel('Y', rotation=0, fontsize=25)
    elif yy == 2:
        ax.set_ylabel('Z', rotation=0, fontsize=25)
    
    # Add text box with the year
    #ax.text(0.05, 0.05, f'Fw attractor, F=1.99', transform=ax.transAxes,
    #        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=17)
    
    #xticks size
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)



def lorenz_A(t, y): #non-autonomous case
    
    #tau = 73 #one year is 73 units as one unit is 5 days
    #slope = 0

    F = 1.99
    dy = [-y[1]**2 - y[2]**2 - a * y[0] + a * F,
          y[0] * y[1] - b * y[0] * y[2] - y[1] + G,
          b * y[0] * y[1] + y[0] * y[2] - y[2]]
    return dy

a = 0.25
b = 4.0
G = 1.0
F0 = 7.0
dt = 0.025

num_trajectories_pb = 50000
forward_initial_conditions = np.random.uniform(-3, 3, size=(num_trajectories_pb, 3))




years = 2
T = 73 * years
num_time_pts = int(T / dt)

#forward trajectories for year 110
t = np.linspace(0, T, num_time_pts)
forward_trajectories = []

for i in trange(num_trajectories_pb):
    initial_conditions = forward_initial_conditions[i, :]
    solution = solve_ivp(lorenz_A, (0, T), initial_conditions, t_eval=t)
    forward_trajectories.append(solution.y)
    
forward_trajectories = np.array(forward_trajectories)
forward_F8 = forward_trajectories[:, :, -20:]


#do the collaps
forward_trajectories_last_collapsed = np.swapaxes(forward_F8, 0, 1)
forward_trajectories_last_collapsed = np.reshape(forward_trajectories_last_collapsed, (forward_trajectories_last_collapsed.shape[0], -1))
forward_trajectories_last_collapsed.shape


fig, ax = plt.subplots(figsize=(10, 10))
#set xlim and y lim to -1, 1
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
heatmaps_plot(ax, forward_trajectories_last_collapsed[:,:], 1, 2, 8.0)
# %%
# forward_F8 = forward_trajectories[:, :, -10:]


# #do the collaps
# forward_trajectories_last_collapsed = np.swapaxes(forward_F8, 0, 1)
# forward_trajectories_last_collapsed = np.reshape(forward_trajectories_last_collapsed, (forward_trajectories_last_collapsed.shape[0], -1))
# forward_trajectories_last_collapsed.shape


# fig, ax = plt.subplots(figsize=(10, 10))
# #set xlim and y lim to -1, 1
# # ax.set_xlim(-1, 1)
# # ax.set_ylim(-1, 1)
# heatmaps_plot(ax, forward_trajectories_last_collapsed[:,:], 1, 2, 8.0)
# %%
fig.savefig("Forward_with_arrow_finale.png", dpi=300)
# %%
