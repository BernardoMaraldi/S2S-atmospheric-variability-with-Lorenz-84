using DynamicalSystems, OrdinaryDiffEq, Attractors, CairoMakie
using Colors  # This is helpful for defining custom colors
using ColorSchemes

ds = Systems.lorenz84(rand(3), F=6, G=1., a=0.25, b=4.0)
xg = range(-3, 4, length=150)
yg = range(-4.0, 3.0; length=150)
zg = range(-1.5, 1.5; length=30)
mapper = AttractorsViaRecurrences(ds, (xg, yg, zg); sparse = false)
basins, attractors = basins_of_attraction(mapper)

z_index = 15  # Choose a z-slice, this is an arbitrary choice for visualization
xy_basins = reshape(basins[:, :, z_index], (length(xg), length(yg)))

fig = Figure(resolution = (400, 400))  # Create a square figure
ax = Axis(fig[1, 1], aspect = DataAspect(), title="", xlabel="X", ylabel="Y")  # Set aspect ratio to data aspect

# Increase labels font size
ax.xlabelsize = 22
ax.ylabelsize = 22
ax.ylabelrotation = -(3.14)*2
#set xlim to -2:4 and ylim to -4:2
#xlims!(ax, -1,4)
#ylims!(ax, -3,2)

#heatmap!(ax, xg, yg, xy_basins)
#colorblind_friendly_colors = cgrad([RGB(0, 0/255, 0/255), RGB(213/255,0/255, 0), RGB(240/255, 228/255, 66/255)], 3)

# Reverse the colormap
reversed_colormap = reverse(cgrad(:lighttest))

# Apply the reversed viridis colormap to the heatmap
heatmap!(ax, xg, yg, xy_basins, colormap=:lighttest)