
# This file produces a figure that shows how particles perform uncertainty propagation, and compares the result to uncertainty propagation through linearization like Measurements.jl does.

using MonteCarloMeasurements, StatsPlots, NNlib, Measurements, KernelDensity
using Measurements: value, uncertainty
default(lab="")
N    = 20
f    = x -> σ(12x-6)
l    = 0
r    = 1
d    = Normal(0.5, 0.15)
m    = Measurements.:(±)(d.μ, d.σ)
my   = f(m)
dm   = Normal(value(my), uncertainty(my))
x    = Particles(N,d).particles |> sort
xr   = LinRange(l,r,100)
y    = f.(x)
noll = zeros(N)


plot(f, l, r, legend=:right, xlims=(l,r), ylims=(l,r), axis=false, grid=false, lab="f(x)", xlabel="Input space", ylabel="Output space")

plot!(x->0.2pdf(d,x),l,r, lab="Input dens.")

kdt = kde(f.(rand(d,100000)), npoints=200, bandwidth=0.08)
plot!(l .+ 0.2kdt.density, kdt.x, lab="True output dens.")

plot!(l .+ 0.2pdf.(Ref(dm),xr), xr, lab="Measurements dens.")

kd = kde(y, npoints=200, bandwidth=0.08)
plot!(l .+ 0.2kd.density, kd.x, lab="Particle kernel dens. est.")

# Draw helper lines
plot!([x x][1:2:end,:]', [noll y][1:2:end,:]', l=(:black, :arrow, :dash, 0.1))
plot!([x fill(l,N).+0.02][1:2:end,:]', [y y][1:2:end,:]', l=(:black, :arrow, :dash, 0.1))

scatter!(x, 0y, lab="Input particles")
scatter!(fill(l,N) .+ 0.02, y, lab="Output particles")

# Draw mean lines
plot!([d.μ,d.μ], [0,f(d.μ)], l=(:red, :dash, 0.2))
plot!([l,d.μ], [f(d.μ),f(d.μ)], l=(:red, :dash, 0.2))
