#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import trange

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

#changes depending on the pullback
years = 25
T = 73 * years
num_time_pts = int(T / dt)
#num_time_pts = 76406

num_trajectories_pb = 10000
pullback_initial_conditions = np.random.uniform(-3, 3, size=(num_trajectories_pb, 3))



#pullback trajectories for year 110
t = np.linspace(73*85, 73*85 + T, num_time_pts)
pullback_trajectories_110_25 = []

for i in trange(num_trajectories_pb):
    initial_conditions = pullback_initial_conditions[i, :]
    solution = solve_ivp(lorenz_NA, (73*85, 73*85+T), initial_conditions, t_eval=t)
    pullback_trajectories_110_25.append(solution.y)

pullback_trajectories_110_25 = np.array(pullback_trajectories_110_25)

#now do the same but for years=30
years = 30
T = 73 * years
num_time_pts = int(T / dt)

#pullback trajectories for year 110
t = np.linspace(73*80, 73*80 + T, num_time_pts)
pullback_trajectories_110_30 = []

for i in trange(num_trajectories_pb):
    initial_conditions = pullback_initial_conditions[i, :]
    solution = solve_ivp(lorenz_NA, (73*80, 73*80+T), initial_conditions, t_eval=t)
    pullback_trajectories_110_30.append(solution.y)

pullback_trajectories_110_30 = np.array(pullback_trajectories_110_30)



#now for years=35
years = 35
T = 73 * years
num_time_pts = int(T / dt)


#pullback trajectories for year 110
t = np.linspace(73*75, 73*75 + T, num_time_pts)
pullback_trajectories_110_35 = []

for i in trange(num_trajectories_pb):
    initial_conditions = pullback_initial_conditions[i, :]
    solution = solve_ivp(lorenz_NA, (73*75, 73*75+T), initial_conditions, t_eval=t)
    pullback_trajectories_110_35.append(solution.y)

pullback_trajectories_110_35 = np.array(pullback_trajectories_110_35)


#now for years 110
years = 110
T = 73 * years
num_time_pts = int(T / dt)

#pullback trajectories for year 110
t = np.linspace(0, T, num_time_pts)
pullback_trajectories_110_110 = []

for i in trange(num_trajectories_pb):
    initial_conditions = pullback_initial_conditions[i, :]
    solution = solve_ivp(lorenz_NA, (0, T), initial_conditions, t_eval=t)
    pullback_trajectories_110_110.append(solution.y)
    
pullback_trajectories_110_110 = np.array(pullback_trajectories_110_110)

#now compute the wasserstein distance for Y^2 + Z^2 distribution between reference realisation (110)
#and the pullback trajectories for years 25, 30, 35

import numpy as np
from scipy.stats import wasserstein_distance

def compute_wasserstein_distances(data_array, comparison_array, pullback_years, dt, N_bins):
    years = []
    wasserstein_distances = []
    #last element of the pullback_years is anni
    anni = pullback_years[-1]+1
    offset = 110-anni
    #offset = 0
    
    for year in pullback_years:
        # Extract the x component of the current year's distribution
        data_year = data_array[:, :, int((36 + 73 * year) / dt):int((36 + 73 * year) / dt) + 240]
        data_year_collapsed = np.swapaxes(data_year, 0, 1)
        data_year_collapsed = np.reshape(data_year_collapsed, (data_year_collapsed.shape[0], -1))
        x_year = data_year_collapsed[1, :]**2 + data_year_collapsed[2, :]**2
        #x_year = data_year_collapsed[0, :]

        comparison_year = comparison_array[:, :, int((36 + 73 * (offset+year)) / dt):int((36 + 73 * (offset+year)) / dt) + 240]
        comparison_year_collapsed = np.swapaxes(comparison_year, 0, 1)
        comparison_year_collapsed = np.reshape(comparison_year_collapsed, (comparison_year_collapsed.shape[0], -1))
        x_comparison_year = comparison_year_collapsed[1, :]**2 + comparison_year_collapsed[2, :]**2
        #x_comparison_year = comparison_year_collapsed[0, :]

        # Compute histogram of the current year's x component
        hist_year, _ = np.histogram(x_year, bins=N_bins, density=True)
        hist_comparison_year, _ = np.histogram(x_comparison_year, bins=N_bins, density=True)

        # Compute the Wasserstein distance
        wasserstein_dist = wasserstein_distance(hist_year, hist_comparison_year)

        # Store the year and Wasserstein distance
        years.append(year)
        wasserstein_distances.append(wasserstein_dist)

    return years, wasserstein_distances

dt = 0.025

# Assuming pullback_trajectories_110_35, pullback_trajectories_110_110, and years_range_35 are defined
years_35, wasserstein_distances_35 = compute_wasserstein_distances(pullback_trajectories_110_35, pullback_trajectories_110_110, range(0, 35), dt, 500)
years_35 = [year+75 for year in years_35]

# Assuming pullback_trajectories_110_30, pullback_trajectories_110_110, and years_range_30 are defined
years_30, wasserstein_distances_30 = compute_wasserstein_distances(pullback_trajectories_110_30, pullback_trajectories_110_110, range(0, 30), dt, 500)
years_30 = [year+80 for year in years_30]

# Assuming pullback_trajectories_110_25, pullback_trajectories_110_110, and years_range_25 are defined
years_25, wasserstein_distances_25 = compute_wasserstein_distances(pullback_trajectories_110_25, pullback_trajectories_110_110, range(0, 25), dt, 500)
years_25 = [year+85 for year in years_25]

# Plot the Wasserstein distances over the years
plt.plot(years_35, wasserstein_distances_35, marker='o', linestyle='-', label='pullback 35', lw=1, ms=3)
plt.plot(years_30, wasserstein_distances_30, marker='o', linestyle='-', label='pullback 30', lw=1, ms=3)
plt.plot(years_25, wasserstein_distances_25, marker='o', linestyle='-', label='pullback 25', lw=1, ms=3)
plt.xlabel('Year')
plt.ylabel('Wasserstein Distance')
plt.title('WS between Each Year and Last Year (500 bins) - X values')
plt.xlim([74, 90])
plt.ylim([0, 0.2])
plt.grid(True)
plt.legend()
plt.show()