# Intraseasonal-atmospheric-variability-with-Lorenz-84
This repository contains the scripts used to create the figures of the paper (arXiv &/or Chaos URL). 
All the information is provided in the code, although refer to the paper (arXiv &/or Chaos URL) for further details.
Here you will find the code to:
  - integrate the 3-ODE Lorenz (1984) model (L84) along a single trajectory, as well as for an ensemble of initial conditions;
  - compute the heatmaps or individual snapshots for the different cases -- autonomous, nonautonomous with seasonal focing only, nonautonomous with both seasonal and climate trend forcing;
  - compute the statistics of the evolving attractor;
  - carry out bifurcation analysis using BifurcationKit.jl;
  - obtain the remaining timeseries and figures.

When a large ensemble of N trajectories is required, the user will have to generate the data by integrating the model equations; this might take several hours on a laptop for 10000 or 50000 trajectories, as used in the paper.
  
