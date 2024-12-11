using ChaosTools, CairoMakie, DynamicalSystems

# Define the Lorenz '84 system rule
function lorenz84_rule(u, p, t)
    a, F, G, b = p
    du1 = -u[2]^2 - u[3]^2 - a * u[1] + a * F
    du2 = u[1] * u[2] - b * u[1] * u[3] - u[2] + G
    du3 = b * u[1] * u[2] + u[1] * u[3] - u[3]
    return SVector{3}(du1, du2, du3)
end

# Initialize the initial state and other fixed parameters
initial_state = fill(1.0, 3)         # Initial state [x, y, z]
fixed_params = [0.0, 6.0, 1.0, 4.0] # Initial parameters [a, F, G, b] with F = 0.0

# Define the range for F and prepare to store the highest Lyapunov exponent
as = 0.0:0.01:1.0
max_λs = zeros(length(as))

# Compute the highest Lyapunov exponent for each F value
for (i, a) in enumerate(as)
    params = [a, fixed_params[2], fixed_params[3], fixed_params[4]]
    lorenz84_system = CoupledODEs(lorenz84_rule, initial_state, params)
    lyap_exponents = lyapunovspectrum(lorenz84_system, 10000; Δt = 0.1)
    max_λs[i] = maximum(lyap_exponents)
end

# Plot the results
fig = Figure()
ax = Axis(fig[1,1]; xlabel = L"a", ylabel = L"\lambda_{max}")
lines!(ax, as, max_λs)
fig

# Increase labels font size
ax.xlabelsize = 20
ax.ylabelsize = 20
ax.ylabelrotation = -(3.14)*2


ax.xticks = 0:0.2:1

fig