from scipy.stats import moment
import numpy as np
import matplotlib.pyplot as plt

# Assuming your reshaped_data has the shape (110, N*240, 3) where 240 is the number of points in one month
# Assuming the variables are ordered as X, Y, Z
# Assuming you have a .npy file with the trajectories (for the statistics computed here
    # a number N=10 trajcetories should still reproduce the results)
    
#this code also produces a heatmap of the attractor (snaphot of an arbitrary year)
#but in order to have a better visualization of the attractor, a larger number of trajectories is needed

#open the file calles concatenated_negative_january.npy which contains the 
reshaped_data = np.load('concatenated_positive_januaries.npy')
reshaped_data = reshaped_data.transpose(0, 1, 3, 2).reshape(reshaped_data.shape[0], -1, reshaped_data.shape[2])

import copy
import matplotlib.colors as colors


my_cmap = copy.copy(plt.cm.get_cmap('hot')) # copy the default cmap
my_cmap.set_bad((0,0,0))


def heatmaps_plot(ax, anno, xx, yy, year):
    heatmap_p = np.histogram2d(
        anno[:, xx],
        anno[:, yy],
        bins=(600, 600),
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
        ax.set_xlabel('X', rotation=0, fontsize=20)
    elif xx == 1:
        ax.set_xlabel('Y', rotation=0, fontsize=20)
    elif xx == 2:
        ax.set_xlabel('Z', rotation=0, fontsize=20)
    
    if yy == 0:
        ax.set_ylabel('X', rotation=0, fontsize=20)
    elif yy == 1:
        ax.set_ylabel('Y', rotation=0, fontsize=20)
    elif yy == 2:
        ax.set_ylabel('Z', rotation=0, fontsize=20)
    
    # Add text box with the year
    ax.text(0.05, 0.05, f'Climate change year: {year}', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=15)
    
    #xticks size
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

#heatmap of the attractor projection, year 110
fig, ax = plt.subplots(figsize=(10, 10))
heatmaps_plot(ax, reshaped_data[109, :, :], 1, 2, 109)

#%%
num_years, num_points, num_variables = reshaped_data.shape

start_year = 10
years_to_plot = np.arange(0, num_years-start_year) + 1

# Initialize arrays to store the statistics
mean_values = np.zeros((num_years, num_variables))
std_values = np.zeros((num_years, num_variables))
skewness_values = np.zeros((num_years, num_variables))
kurtosis_values = np.zeros((num_years, num_variables))

# Compute the statistics for each year and each variable
for year in range(num_years):
    for variable in range(num_variables):
        data_slice = reshaped_data[year, :, variable]
        mean_values[year, variable] = np.mean(data_slice)
        std_values[year, variable] = np.std(data_slice)
        skewness_values[year, variable] = moment(data_slice, moment=3)
        kurtosis_values[year, variable] = moment(data_slice, moment=4)

# Plotting
years = np.arange(110) + 1  # Assuming years start from 1
plt.figure(figsize=(10, 6))

colors = ['#000000', '#E69F00', '#56B4E9', '#009E73']

plt.plot(years_to_plot, mean_values[start_year:, 0], label='Mean', c=colors[0])
plt.plot(years_to_plot, std_values[start_year:, 0], label='Standard Deviation', c=colors[1])
plt.plot(years_to_plot, skewness_values[start_year:, 0], label='Skewness', c=colors[2])
plt.plot(years_to_plot, kurtosis_values[start_year:, 0], label='Kurtosis', c=colors[3])

plt.xlabel('t [years]', fontsize=20)
#plt.ylabel('Moment Value')
#plt.title('Statistical Moments for Variable x Over Time')
#plt.legend(fontsize=20)
plt.grid(True)
#ticks fontsize
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('stats_january_positive_X.png', dpi=400)

plt.show()
#save the figure stats_january negative


squared_sum_yz = reshaped_data[:, :, 1]**2 + reshaped_data[:, :, 2]**2

# Exclude the first 10 years
start_year = 10
years_to_plot = np.arange(0, num_years-start_year) + 1  # Assuming years start from 1

# Initialize arrays to store the statistics
mean_values_yz = np.zeros((num_years,))
std_values_yz = np.zeros((num_years,))
skewness_values_yz = np.zeros((num_years,))
kurtosis_values_yz = np.zeros((num_years,))

# Compute the statistics for each year for the squared sum of y and z
for year in range(num_years):
    data_slice_yz = squared_sum_yz[year, :]
    mean_values_yz[year] = np.mean(data_slice_yz)
    std_values_yz[year] = np.std(data_slice_yz)
    skewness_values_yz[year] = moment(data_slice_yz, moment=3)
    kurtosis_values_yz[year] = moment(data_slice_yz, moment=4)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(years_to_plot, mean_values_yz[start_year:], label='Mean', c=colors[0])
plt.plot(years_to_plot, std_values_yz[start_year:], label='Standard Deviation', c=colors[1])
plt.plot(years_to_plot, skewness_values_yz[start_year:], label='Skewness', c=colors[2])
plt.plot(years_to_plot, kurtosis_values_yz[start_year:], label='Kurtosis', c=colors[3])

plt.xlabel('t [years]', fontsize=20)
#plt.ylabel('M')
#plt.title('Statistical Moments for Squared Sum of y and z Over Time (Excluding First 10 Years)')
#plt.legend(fontisize=20)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#save the figure stats_january negative
plt.savefig('stats_january_positive_YZ.png', dpi=400)

plt.show()