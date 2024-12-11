#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #I use this for the integration, RK4

# Define the differential equation dx/dt = -Ax + B*sin(t)
def differential_eq(t, x):
    return -A * x + B * np.sin(t)

# Parameters
A = 0.3  # Example value for A
B = 1.0  # Example value for B
x0 = 2.3  # Initial condition

# Time spans
t0_list = [0.0, 1.0, 2.0]  # Three different pullback times
t_end = 10.0  # End time for integration
tempo = np.linspace(0, t_end, 500) 

t0_list = np.linspace(-10,0,5) 
soluzioni=[]
#colormap
cmap = plt.get_cmap('copper')

# Plotting
plt.figure(figsize=(10, 6))
for t0 in t0_list:
    # Integrate the differential equation
    tempo = np.linspace(t0, t_end, 500)
    sol = solve_ivp(differential_eq, (t0, t_end), (x0, t0), t_eval=tempo) 
    #soluzioni.append(sol).y[0]
    # Plot the result
    plt.plot(sol.t, sol.y[0], label=f"s = {t0}", color=cmap((t0+10)/10))

# Customize the plot
plt.xlabel('Time', fontsize=26)
plt.ylabel('x(t)', rotation=0, fontsize=26)
plt.legend(fontsize=16)
plt.grid(True)
#ticks size
plt.tick_params(axis='both', which='major', labelsize=18)

# Show the plot
plt.show()


# %%
