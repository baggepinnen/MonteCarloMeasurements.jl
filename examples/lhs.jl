# This script illustrates how to use latin hypercube sampling. In the first example, we produce a sample with a non-diagonal covariance matrix to illustrate that the latin property is lost for all dimensions but the first:
using MonteCarloMeasurements, LatinHypercubeSampling, Test
ndims  = 2
N      = 40   # Number of particles
ngen   = 2000 # How long to run optimization
X, fit = LHCoptim(N,ndims,ngen)
m, Σ   = [1,2], [2 1; 1 4] # Desired mean and covariance
particles = transform_moments(X, m, Σ)
@test mean(particles, dims=1)[:] ≈ m
@test cov(particles) ≈ Σ

p = Particles(particles)
plot(scatter(eachcol(particles)..., title="Sample"), plot(fit, title="Fitness vs. iteration"))
vline!(particles[:,1]) # First dimension is still latin
hline!(particles[:,2]) # Second dimension is not


# If we do the same thing with a diagonal covariance matrix, the latin property is approximately preserved in all dimensions provided that the latin optimizer was run sufficiently long.
m, Σ   = [1,2], [2 0; 0 4] # Desired mean and covariance
particles = transform_moments(X, m, Σ)
p         = Particles(particles)
plot(scatter(eachcol(particles)..., title="Sample"), plot(fit, title="Fitness vs. iteration"))
vline!(particles[:,1]) # First dimension is still latin
hline!(particles[:,2]) # Second dimension is not

# We provide a method which absolutely preserves the latin property in all dimensions, but if you use this, the covariance of the sample will be slighlty wrong
particles = transform_moments(X, m, Σ, preserve_latin=true)
p         = Particles(particles)
plot(scatter(eachcol(particles)..., title="Sample"), plot(fit, title="Fitness vs. iteration"))
vline!(particles[:,1]) # First dimension is still latin
hline!(particles[:,2]) # Second dimension is not
@test mean(particles, dims=1)[:] ≈ m
@test cov(particles) ≉ Σ # OBS not the not approx equal!

# We can also visualize the statistics of the sample
using StatsPlots
corrplot(particles)
#
plot(density(p[1]), density(p[2]))
