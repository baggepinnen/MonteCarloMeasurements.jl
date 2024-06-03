# Examples
## [Control systems](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/controlsystems.jl)
This example shows how to simulate control systems (using [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl)) with uncertain parameters. We calculate and display Bode diagrams, Nyquist diagrams and time-domain responses. We also illustrate how the package [ControlSystemIdentification.jl](https://github.com/baggepinnen/ControlSystemIdentification.jl) interacts with MonteCarloMeasurements to facilitate the creation and analysis of uncertain systems.

We also perform some limited benchmarks.

The package [RobustAndOptimalControl.jl](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/) contains lots of additional tools to work with linear systems with uncertainty represented as `Particles`. See the documentation on [Uncertainty modeling](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/uncertainty/) for several examples.

## [Latin Hypercube Sampling](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/lhs.jl)
We show how to initialize particles with LHS and how to make sure the sample gets the desired moments. We also visualize the statistics of the sample.

## [How MC uncertainty propagation works](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/transformed_densities.jl)
We produce the first figure in this readme and explain in visual detail how different forms of uncertainty propagation propagates a probability distribution through a nonlinear function. Also see [Comparison between linear uncertainty propagation and Monte-Carlo sampling](@ref) for more visual examples.

## [Robust probabilistic optimization](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/robust_controller_opt.jl)
Here, we use MonteCarloMeasurements to perform [robust optimization](https://en.wikipedia.org/wiki/Robust_optimization). With robust and probabilistic, we mean that we place some kind of bound on a quantile of an uncertain value, or otherwise make use of the probability distribution of some value that depend on the optimized parameters.

The application we consider is optimization of a PID controller. Normally, we are interested in controller performance and robustness against uncertainty. The robustness is often introduced by placing an upper bound on the, so called, sensitivity function. When the system to be controlled is parameterized by `Particles`, we can penalize both variance in the performance measure as well as the 90:th quantile of the maximum of the sensitivity function. This example illustrates how easy it is to incorporate probabilistic constrains or cost functions in an optimization problem using `Particles`.


## [Autodiff and Robust optimization](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/autodiff_robust_opt.jl)
Another example using MonteCarloMeasurements to perform [robust optimization](https://en.wikipedia.org/wiki/Robust_optimization), this time with automatic differentiation. We use Optim.jl to solve a linear program with probabilistic constraints using 4 different methods, two gradient free, one first-order and one second-order method. We demonstrate calculation of gradients of uncertain functions with uncertain inputs using both Zygote.jl and ForwardDiff.jl.

## Unitful interaction
Particles with units can be created using the package [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) for uncertainty propagation with automatic unit checks. The interaction is only supported through the construct `Particles{Quantity}`, whereas the reverse construct `Quantity{Particles}` is likely to result in problems. Unitful particles are thus created like this
```@repl
using MonteCarloMeasurements, Unitful
(1 ± 0.1)u"V"
(1..2)u"m"
```

### Example: Solar collector energy transfer
The following example estimates the amount of instantaneous thermal power transferred from a solar collector embedded in a concrete floor, to a water reservoir. The power is computed by an measuring the temperature difference, $\Delta T$, between the warm water going into the collector tank and the colder water returning. Using the mass-flow rate and the specific heat capacity of water, we can estimate the power transfer. No flow meter is installed, so the flow is estimated and subject to large uncertainty.
```@example solar
using MonteCarloMeasurements
using Unitful
using Unitful: W, kW, m, mm, hr, K, g, J, l, s

ΔT = (3.5 ± 0.8)K # The temperature difference varies slightly between different flow circuits.
specific_heat_water = 4.19J/(g*K)
density_water = 1e6g/m^3
flow = 8*(1..2.5)*l/(60s) # 8 solar collector circuits, each with an estimated flow rate of between 1 and 2.5 l/minute
mass_flow = flow * density_water |> upreferred # Water flow in mass per second
power = uconvert(W, mass_flow * specific_heat_water * ΔT) # Power in Watts
```

Some power is lost to the ground in which the heat-exchanger circuits are embedded, we estimate this to be between 10 and 50% of the total power.
```@example solar
ground_losses = (0.1..0.5) * power # Between 10-50% power loss to ground
reservoir_volume = 7m*3m*1.5m
```

The energy transfered during 6hrs solar collection can be estimated as
```@example solar
energy_6_hrs = (power - ground_losses)*6hr
```
and this energy transfer will increase the temperature in the reservoir by
```@example solar
ΔT_reservoir_6hr = energy_6_hrs/(reservoir_volume*density_water*specific_heat_water) |> upreferred
```

Finally, we visualize the distributions associated with the estimated quantities:
```@example solar
using Plots
figh = plot(ΔT_reservoir_6hr, xlabel="\$ΔT [K]\$", ylabel="\$P(ΔT)\$", title="Temperature increase 6hrs sun")
qs   = 0:0.01:1
Qs   = pquantile.(ΔT_reservoir_6hr, qs)
figq = plot(qs, Qs, xlabel="∫\$P(ΔT)\$")
plot(figh, figq)
```


## Monte-Carlo sampling properties
The variance introduced by Monte-Carlo sampling has some fortunate and some unfortunate properties. It decreases as 1/N, where N is the number of particles/samples. This unfortunately means that to get half the standard deviation in your estimate, you need to quadruple the number of particles. On the other hand, this variance does not depend on the dimension of the space, which is very fortunate.

In this package, we perform [*systematic sampling*](https://arxiv.org/pdf/cs/0507025.pdf) whenever possible. This approach exhibits lower variance than standard random sampling. Below, we investigate the variance of the mean estimator of a random sample from the normal distribution. The variance of the estimate of the mean is known to decrease as 1/N
```julia
default(l=(3,))
N = 1000
svec = round.(Int, exp10.(LinRange(1, 3, 50)))
vars = map(svec) do i
  var(mean(randn(i)) for _ in 1:1000)
end
plot(svec, vars, yscale=:log10, xscale=:log10, lab="Random sampling", xlabel="\$N\$", ylabel="Variance")
plot!(svec, N->1/N, lab="\$1/N\$", l=(:dash,))
vars = map(svec) do i
  var(mean(systematic_sample(i)) for _ in 1:1000)
end
plot!(svec, vars, lab="Systematic sampling")
plot!(svec, N->1/N^2, lab="\$1/N^2\$", l=(:dash,))
```
![variance plot](assets/variance.svg)

As we can see, the variance of the standard random sampling decreases as expected. We also see that the variance for the systematic sample is considerably lower, and also scales as (almost) 1/N².

A simplified implementation of the systematic sampler is given below
```julia
function systematic_sample(N, d=Normal(0,1))
    e   = rand()/N
    y   = e:1/N:1
    map(x->quantile(d,x), y)
end
```
~~As we can see, a single random number is generated to seed the entire sample.~~ (This has been changed to `e=0.5/N` to have a correct mean.) The samples are then drawn deterministically from the quantile function of the distribution.

## Variational inference
See [blog post](https://cscherrer.github.io/post/variational-importance-sampling/) by [@cscherrer](https://github.com/cscherrer) for an example of variational inference using `Particles`




## Differential Equations
[The tutorial](http://tutorials.juliadiffeq.org/html/type_handling/02-uncertainties.html) for solving differential equations using `Measurement` works for `Particles` as well. A word of caution for actually using Measurements.jl in this example: while solving the pendulum on short time scales, linear uncertainty propagation works well, as evidenced by the below simulation of a pendulum with uncertain properties
```julia
function sim(±, tspan, plotfun=plot!; kwargs...)
    g = 9.79 ± 0.02; # Gravitational constant
    L = 1.00 ± 0.01; # Length of the pendulum
    u₀ = [0 ± 0, π / 3 ± 0.02] # Initial speed and initial angle

    #Define the problem
    function simplependulum(du,u,p,t)
        θ  = u[1]
        dθ = u[2]
        du[1] = dθ
        du[2] = -(g/L) * sin(θ)
    end

    prob = ODEProblem(simplependulum, u₀, tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-6)

    plotfun(sol.t, getindex.(sol.u, 2); kwargs...)
end

tspan = (0.0, 5)
plot()
sim(Measurements.:±, tspan, label = "Linear", xlims=(tspan[2]-5,tspan[2]))
sim(MonteCarloMeasurements.:±, tspan, label = "MonteCarlo", xlims=(tspan[2]-5,tspan[2]))
```
![window](assets/short_timescale.svg)

The mean and errorbars for both Measurements and MonteCarloMeasurements line up perfectly when integrating over 5 seconds.

However, the uncertainty in the pendulum coefficients implies that the frequency of the pendulum oscillation is uncertain, when solving on longer time scales, this should result in the phase being completely unknown, something linear uncertainty propagation does not handle
```julia
tspan = (0.0, 200)
plot()
sim(Measurements.:±, tspan, label = "Linear", xlims=(tspan[2]-5,tspan[2]))
sim(MonteCarloMeasurements.:±, tspan, label = "MonteCarlo", xlims=(tspan[2]-5,tspan[2]))
```
![window](assets/long_timescale.svg)

We now integrated over 200 seconds and look at the last 5 seconds. This result maybe looks a bit confusing, the linear uncertainty propagation is very sure about the amplitude at certain points but not at others, whereas the Monte-Carlo approach is completely unsure. Furthermore, the linear approach thinks that the amplitude at some points is actually much higher than the starting amplitude, implying that energy somehow has been added to the system! The picture might become a bit more clear by plotting the individual trajectories of the particles
```julia
plot()
sim(Measurements.:±, tspan, label = "Linear", xlims=(tspan[2]-5,tspan[2]), l=(5,))
sim(MonteCarloMeasurements.:∓, tspan, mcplot!, label = "", xlims=(tspan[2]-5,tspan[2]), l=(:black,0.1))
```
![window](assets/long_timescale_mc.svg)

It now becomes clear that each trajectory has a constant amplitude (although individual trajectories amplitudes vary slightly due to the uncertainty in the initial angle), but the phase is all mixed up due to the slightly different frequencies!

These problems grow with increasing uncertainty and increasing integration time. In fact, the uncertainty reported by Measurements.jl goes to infinity as the integration time does the same.

Of course, the added accuracy from using MonteCarloMeasurements does not come for free, as it costs some additional computation. We have the following timings for integrating the above system 100 seconds using three different uncertainty representations
```julia
Measurements.:±             14.596 ms  (729431 allocations: 32.43 MiB)   # Measurements.Measurement
MonteCarloMeasurements.:∓   25.115 ms  (25788 allocations: 24.68 MiB)    # 100 StaticParticles
MonteCarloMeasurements.:±   345.730 ms (696212 allocations: 838.50 MiB)  # 500 Particles
```

# MCMC inference using Turing.jl

[Turing.jl](https://github.com/TuringLang/Turing.jl/) is a probabilistic programming language, and an interface between Turing and MonteCarloMeasurements is provided by
[Turing2MonteCarloMeasurements.jl](https://github.com/baggepinnen/Turing2MonteCarloMeasurements.jl) with instructions and examples in the readme.
