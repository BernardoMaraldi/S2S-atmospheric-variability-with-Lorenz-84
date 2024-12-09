# Intraseasonal-atmospheric-variability-with-Lorenz-84
This repository contains the scripts used to create the figures of the paper (). 
All the information is provided in the code, although refer to the paper () for further details.
Here you will find the code to:
  - integrate the 3-equation Lorenz 1984 model (L84) along one single trajectory and for an ensemble of initial conditions;
  - compute the heatmaps (snapshots), for the different cases (autonomous, nonautonomous seasonal focing, nonautonomous seasonal+trend forcing);
  - compute the statistics of the evolving attractor;
  - carry out bifurcation analysis (with BifurcationKit.jl);
  - obtain the various remaining figures (timeseries).

When a large ensemble is used, the user will have to generate the data by integrating the equations (which might take up to 10 hours with 50000 trajectories)
  
