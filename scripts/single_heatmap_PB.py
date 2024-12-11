#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #I 
import copy
import matplotlib.colors as colors
#trange
from tqdm import trange

my_cmap = copy.copy(plt.cm.get_cmap('hot')) # copy the default cmap
my_cmap.set_bad((0,0,0))

def heatmaps_plot(ax, anno, xx, yy):
    heatmap_p = np.histogram2d(
        anno[xx, :],
        anno[yy, :],
        bins=(500, 500),
        range=[[-2.5, 2.5], [-2.5, 2.5]]
    )[0]
    
    extent = [-2.5, 2.5, -2.5, 2.5]


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



def lorenz_NA(t, y): #non-autonomous case
    
    tau = 73 #one year is 73 units as one unit is 5 days
   
    F0 = 7
    
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

num_trajectories_pb = 50
pullback_initial_conditions = np.random.uniform(-3, 3, size=(num_trajectories_pb, 3))

years = 6
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
#here you can select the required snapshot by choosing the right index
#for F = 6, the index is 1944, so we select the 1944th element of the 6th year 
#each year has 2920 elements
#for F = 8 select 485:487 + 5*2920. 
#this is found by dividing the unit at which F=8 by the timestep 0.025
pullback_F6 = pullback_trajectories[:, :, 1944+5*2920:1946+5*2920] 


#do the collaps
pullback_trajectories_last_collapsed = np.swapaxes(pullback_F6, 0, 1)
pullback_trajectories_last_collapsed = np.reshape(pullback_trajectories_last_collapsed, (pullback_trajectories_last_collapsed.shape[0], -1))
pullback_trajectories_last_collapsed.shape


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
#set xlim and y lim to -1, 1
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
ax.set_xlabel('Y', fontsize=15)
ax.set_ylabel('Z', rotation=0, fontsize=15)
heatmaps_plot(ax, pullback_trajectories_last_collapsed[:,:], 1, 2)
