# In this example, we will create a transfer function with uncretain coefficients, and use it to calculate bode diagrams and simulate the system.
using ControlSystems, MonteCarloMeasurements, StatsPlots
using Test, LinearAlgebra, Statistics
import MonteCarloMeasurements: ±,⊗
#
p = 1 ± 0.1
ζ = 0.3 ± 0.05
ω = 1 ± 0.1
# p,ζ,ω = outer_product([Normal(1,0.1),Normal(0.3,0.05),Normal(1,0.1)], 2_000) # An alternative to the above
# p,ζ,ω = [1,0.3,1] ⊗ [0.1, 0.05, 0.1] # Another equivalent alternative, but defaults to N≈100_000
G = tf([p*ω], [1, 2ζ*ω, ω^2])

dc = dcgain(G)[]
density(dc)
w = exp10.(LinRange(-0.7,0.7,100))
@time mag, phase = bode(G,w) .|> vec

scales = (yscale=:log10, xscale=:log10)
errorbarplot(w,mag,0.00; scales..., layout=3, subplot=1, lab="q=$(0.00)")
errorbarplot!(w,mag,0.01, subplot=1, lab="q=$(0.01)")
errorbarplot!(w,mag,0.1, subplot=1, lab="q=$(0.1)", legend=:bottomleft, linewidth=3)
mcplot!(w,mag; scales..., alpha=0.2, subplot=2, c=:black)
ribbonplot!(w,mag, 2; yscale=:log10, xscale=:log10, alpha=0.2, subplot=3)
# A ribbonplot is not always suitable for plots with logarithmic scales.


reny,imny,wny = nyquist(G,w) .|> vec
plot(reny, imny, 0.005, lab="Nyquist curve 99%")
plot!(reny, imny, 0.025, lab="Nyquist curve 95%", ylims=(-4,0.2), xlims=(-2,2), legend=:bottomright)
vline!([-1], l=(:dash, :red), primary=false)
vline!([0], l=(:dash, :black), primary=false)
hline!([0], l=(:dash, :black), primary=false)
##
plot(reny, imny, lab="Nyquist curve 95%", ylims=(-4,0.2), xlims=(-2,2), legend=:bottomright, points=true)
vline!([-1], l=(:dash, :red), primary=false)
vline!([0], l=(:dash, :black), primary=false)
hline!([0], l=(:dash, :black), primary=false)
##

## Time Simulations
Pd = c2d(G, 0.1)

y,t,x = step(Pd, 20)
errorbarplot(t,y[:], 0.00, layout=3, subplot=1, alpha=0.5)
errorbarplot!(t,y[:], 0.05, subplot=1, alpha=0.5)
errorbarplot!(t,y[:], 0.1, subplot=1, alpha=0.5)
mcplot!(t,y[:], subplot=2, l=(:black, 0.02))
ribbonplot!(t,y[:], subplot=3)






## bode benchmark =========================================
# using BenchmarkTools, Printf
# p = 1 ± 0.1
# ζ = 0.3 ± 0.1
# ω = 1 ± 0.1
# G = tf([p*ω], [1, 2ζ*ω, ω^2])
# t1 = @belapsed bode($G,$w)
# p = 1
# ζ = 0.3
# ω = 1
# G = tf([p*ω], [1, 2ζ*ω, ω^2])
# t2 = @belapsed bode($G,$w)
#
# @printf("Time with 500 particles: %16.4fms \nTime with regular floating point: %7.4fms\n500×floating point time: %16.4fms\nSpeedup factor: %22.1fx\n", 1000*t1, 1000*t2, 1000*500t2, 500t2/t1)
