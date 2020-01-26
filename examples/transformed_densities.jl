
# This file produces a figure that shows how particles perform uncertainty propagation, and compares the result to uncertainty propagation through linearization like Measurements.jl does.

using MonteCarloMeasurements, StatsPlots, NNlib, Measurements, KernelDensity
using Measurements: value, uncertainty
default(lab="")
N    = 20 # Number of particles
f    = x -> σ(12x-6) # Nonlinear function
l    = 0 # Left boundary
r    = 1 # Right boundary
d    = Normal(0.5, 0.15) # The probability density of the input
m    = Measurements.:(±)(d.μ, d.σ) # For comparison to Measurements.jl using linear uncertainty propagation
my   = f(m) # output measurement
dm   = Normal(value(my), uncertainty(my)) # Output density according to Measurements
x    = Particles(N,d, permute=false) # Create particles distributed according to d, sort for visualization
y    = f(x).particles # corresponding output particles
x    = x.particles # extract vector to plot manually
xr   = LinRange(l,r,100) # x values for plotting
noll = zeros(N)


plot(f, l, r, legend=:right, xlims=(l,r), ylims=(l,r), axis=false, grid=false, lab="f(x)", xlabel="Input space", ylabel="Output space")
plot!(x->0.2pdf(d,x),l,r, lab="Input dens.")
# Estimate the true output density using a large sample
kdt = kde(f.(rand(d,100000)), npoints=200, bandwidth=0.08)
plot!(l .+ 0.2kdt.density, kdt.x, lab="True output dens.")

# This is the output density as approximated by linear uncertainty propagation
plot!(l .+ 0.2pdf.(Ref(dm),xr), xr, lab="Linear Gaussian propagation")

# Estimate the output density corresponding to the particles
kd = kde(y, npoints=200, bandwidth=0.08)
plot!(l .+ 0.2kd.density, kd.x, lab="Particle kernel dens. est.", l=:dash)

# Draw helper lines that show how particles are transformed from input space to output space
plot!([x x][1:2:end,:]', [noll y][1:2:end,:]', l=(:black, :arrow, :dash, 0.1))
plot!([x fill(l,N).+0.02][1:2:end,:]', [y y][1:2:end,:]', l=(:black, :arrow, :dash, 0.1))

# Plot the particles
scatter!(x, 0y, lab="Input particles")
scatter!(fill(l,N) .+ 0.02, y, lab="Output particles")

# Draw mean lines, these show hoe the mean is transformed using linear uncertainty propagation
plot!([d.μ,d.μ], [0,f(d.μ)], l=(:red, :dash, 0.2))
plot!([l,d.μ], [f(d.μ),f(d.μ)], l=(:red, :dash, 0.2))
