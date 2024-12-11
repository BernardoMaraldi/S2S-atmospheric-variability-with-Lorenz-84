#%% 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #I use this for the integration, RK4



def lorenz_NA(t, y): #non-autonomous case
    F0=7
    tau = 73 #one year is 73 units as one unit is 5 days
    F = F0 + 2 * np.cos(2 * np.pi * t / tau) # seasonal forcing
    dy = [-y[1]**2 - y[2]**2 - a * y[0] + a * F,
          y[0] * y[1] - b * y[0] * y[2] - y[1] + G,
          b * y[0] * y[1] + y[0] * y[2] - y[2]]
    return dy

a = 0.25
b = 4
G = 1

# Set the initial conditions
y0 = [2, 1, 0]
F0 = 3.0
dt = 0.025
years = 20
T = 73 * years
num_time_pts = int(T / dt)

t = np.linspace(0, T, num_time_pts)

lorenz_solution = solve_ivp(lorenz_NA, (0, T), y0, t_eval=t) 
t = lorenz_solution.t
y = lorenz_solution.y.T

#plot the flow X
fig1 = plt.figure(figsize=(12, 6))

ax1 = fig1.add_subplot(111)
ax1.plot(t[-2*2920:], y[-2*2920:,0] )
ax1.set_xticks([int(18*73), int(19*73), int(20*73)])
ax1.set_xticklabels([0, 1, 2])
#increase the font size of the axis labels
ax1.tick_params(axis='both', which='major', labelsize=20)

ax1.set_xlabel('t [years]', fontsize=20)
ax1.set_ylabel('X', fontsize=20, rotation=0)


#plot the energy E=1/2(x^2+y^2+z^2)
fig2 = plt.figure(figsize=(12, 6))

ax2 = fig2.add_subplot(111)

ax2.plot(t[-2*2920:], (y[-2*2920:, 0]**2+y[-2*2920:, 1]**2+y[-2*2920:, 2]**2)/2, label='x', c = 'darkred')

ax2.set_xticks([int(18*73), int(19*73), int(20*73)])
ax2.set_xticklabels([0, 1, 2])
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_xlabel('t [years]', fontsize=20)
ax2.set_ylabel('X', fontsize=20)
ax2.yaxis.set_label_text('$E_{TOT}$', fontsize=20)

# %%
