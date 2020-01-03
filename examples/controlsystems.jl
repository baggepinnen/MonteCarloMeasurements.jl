# # ControlSystems using MonteCarloMeasurements
# In this example, we will create a transfer function with uncretain coefficients, and use it to calculate bode diagrams and simulate the system.
using ControlSystems, MonteCarloMeasurements, StatsPlots
using Test, LinearAlgebra, Statistics
import MonteCarloMeasurements: ⊗
unsafe_comparisons(true, verbose=false) # This file requires mean comparisons for displaying transfer functions in text form as well as for discretizing a LTIsystem
default(size=(2000,1200))

p = 1 ± 0.1
ζ = 0.3 ± 0.05
ω = 1 ± 0.1;
# Alternative definitions of the uncertain parameters are given by

# `p,ζ,ω = outer_product([Normal(1,0.1),Normal(0.3,0.05),Normal(1,0.1)], 2_000)`

# `p,ζ,ω = [1,0.3,1] ⊗ [0.1, 0.05, 0.1] # Defaults to N≈100_000`
G = tf([p*ω], [1, 2ζ*ω, ω^2])

dc = dcgain(G)[]
density(dc)
#
w = exp10.(LinRange(-0.7,0.7,100))
@time mag, phase = bode(G,w) .|> vec;

# ## Bode plot
scales = (yscale=:log10, xscale=:log10)
errorbarplot(w,mag,0.00; scales..., layout=3, subplot=1, lab="q=$(0.00)")
errorbarplot!(w,mag,0.01, subplot=1, lab="q=$(0.01)")
errorbarplot!(w,mag,0.1, subplot=1, lab="q=$(0.1)", legend=:bottomleft, linewidth=3)
mcplot!(w,mag; scales..., alpha=0.2, subplot=2, c=:black)
ribbonplot!(w,mag, 0.95; yscale=:log10, xscale=:log10, alpha=0.2, subplot=3)
# A ribbonplot is not always suitable for plots with logarithmic scales.

# ## Nyquist plot
# We can visualize the uncertainty in the Nyquist plot in a number of different ways, here are two examples
reny,imny,wny = nyquist(G,w) .|> vec
plot(reny, imny, 0.005, lab="Nyquist curve 99%")
plot!(reny, imny, 0.025, lab="Nyquist curve 95%", ylims=(-4,0.2), xlims=(-2,2), legend=:bottomright)
vline!([-1], l=(:dash, :red), primary=false)
vline!([0], l=(:dash, :black), primary=false)
hline!([0], l=(:dash, :black), primary=false)
#
plot(reny, imny, lab="Nyquist curve 95%", ylims=(-4,0.2), xlims=(-2,2), legend=:bottomright, points=true)
vline!([-1], l=(:dash, :red), primary=false)
vline!([0], l=(:dash, :black), primary=false)
hline!([0], l=(:dash, :black), primary=false)
#

# ## Time Simulations
# We start by sampling the system to obtain a discrete-time model.
@unsafe Pd = c2d(G, 0.1)
# We then simulate an plot the results
y,t,x = step(Pd, 20)
errorbarplot(t,y[:], 0.00, layout=3, subplot=1, alpha=0.5)
errorbarplot!(t,y[:], 0.05, subplot=1, alpha=0.5)
errorbarplot!(t,y[:], 0.1, subplot=1, alpha=0.5)
mcplot!(t,y[:], subplot=2, l=(:black, 0.02))
ribbonplot!(t,y[:], subplot=3)

# # System identification
using MonteCarloMeasurements, ControlSystemIdentification, ControlSystems
using Random, LinearAlgebra
# We start by creating a system to use as the subject of identification and some data to use for identification
N  = 500      # Number of time steps
t  = 1:N
Δt = 1        # Sample time
u  = randn(N) # A random control input
G  = tf(0.8, [1,-0.9], 1) # An interesting system
y  = lsim(G,u,t)[1][:]
yn = y + randn(size(y));

# Validation data
uv  = randn(N)
yv  = lsim(G,uv,t)[1][:]
ynv = yv + randn(size(yv));
# Identification parameters
na,nb,nc = 2,1,1

Gls,Σls     = arx(Δt,yn,u,na,nb) # Regular least-squares estimation
Gtls,Σtls   = arx(Δt,yn,u,na,nb, estimator=tls) # Total least-squares estimation
Gwtls,Σwtls = arx(Δt,yn,u,na,nb, estimator=wtls_estimator(y,na,nb)) # Weighted Total least-squares estimation

# Next, we create transfer functions with uncertainty, the first argument is the particle type we want to use
Gls = TransferFunction(Particles, Gls, Σls)
Gtls = TransferFunction(Particles, Gtls, Σtls)
Gwtls = TransferFunction(Particles, Gwtls, Σwtls)

# We now calculate and plot the Bode diagrams for the uncertain transfer functions
scales = (yscale=:log10, xscale=:log10)
w = exp10.(LinRange(-3,log10(π),30))
magG = bode(G,w)[1][:]
mag = bode(Gls,w)[1][:]
errorbarplot(w,mag,0.01; scales..., layout=3, subplot=1, lab="ls")
# plot(w,mag; scales..., layout=3, subplot=1, lab="ls") # src
plot!(w,magG, subplot=1)
mag = bode(Gtls,w)[1][:]
errorbarplot!(w,mag,0.01; scales..., subplot=2, lab="qtls")
# plot!(w,mag; scales..., subplot=2, lab="qtls") # src
plot!(w,magG, subplot=2)
mag = bode(Gwtls,w)[1][:]
errorbarplot!(w,mag,0.01; scales..., subplot=3, lab="wtls")
# plot!(w,mag; scales..., subplot=3, lab="wtls") # src
plot!(w,magG, subplot=3)

## bode benchmark =========================================
using MonteCarloMeasurements, BenchmarkTools, Printf, ControlSystems
using ChangePrecision
@changeprecision Float32 begin
w = exp10.(LinRange(-3,log10(π),30))
p = 1. ± 0.1
ζ = 0.3 ± 0.1
ω = 1. ± 0.1
G = tf([p*ω], [1, 2ζ*ω, ω^2])
t1 = @belapsed bode($G,$w)
p = 1.
ζ = 0.3
ω = 1.
G = tf([p*ω], [1., 2ζ*ω, ω^2])
sleep(0.5)
t2 = @belapsed bode($G,$w)
using Measurements
p = Measurements.:(±)(1., 0.1)
ζ = Measurements.:(±)(0.3, 0.1)
ω = Measurements.:(±)(1., 0.1)
G = tf([p*ω], [1, 2ζ*ω, ω^2])
sleep(0.5)
t3 = @belapsed bode($G,$w)

p = 1. ∓ 0.1
ζ = 0.3 ∓ 0.1
ω = 1. ∓ 0.1
G = tf([p*ω], [1, 2ζ*ω, ω^2])
sleep(0.5)
t4 = @belapsed bode($G,$w)
p,ζ,ω = StaticParticles(sigmapoints([1, 0.3, 1], 0.1^2))
G = tf([p*ω], [1, 2ζ*ω, ω^2])
sleep(0.5)
t5 = @belapsed bode($G,$w)
end
##
@printf("
| Benchmark | Result |
|-----------|--------|
| Time with 500 particles | %16.4fms |
| Time with regular floating point | %7.4fms |
| Time with Measurements | %17.4fms |
| Time with 100 static part. | %13.4fms |
| Time with static sigmapoints. | %10.4fms |
| 500×floating point time | %16.4fms |
| Speedup factor vs. Manual | %11.1fx |
| Slowdown factor vs. Measurements | %4.1fx |
| Slowdown static vs. Measurements | %4.1fx |
| Slowdown sigma vs. Measurements | %5.1fx|\n",
1000*t1, 1000*t2, 1000*t3, 1000*t4, 1000*t5, 1000*500t2, 500t2/t1, t1/t3, t4/t3, t5/t3) #src
