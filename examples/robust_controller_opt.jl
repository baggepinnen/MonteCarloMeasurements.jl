# In this script, we will design a PID controller by optimization. The system model has uncertain parameters, and we pay a price not only for poor performance of the closed-loop system, but also for a high variance in the performance. In addition to this, we place a constraint on the 90:th percentile of the maximum of the sensitivity function. This way, we will get a doubly robust controller as a result :)
# We start by defining the system and some initial controller parameters
using MonteCarloMeasurements, Optim, ControlSystems, Plots
using MonteCarloMeasurements: ∓
p = 1 ∓ 0.1
ζ = 0.3 ∓ 0.05
ω = 1 ∓ 0.1
P = tf([p*ω], [1, 2ζ*ω, ω^2])
w = exp10.(LinRange(-2,2,100))
params = [1,0.1,0.1]
Msc = 1.5 # Constrint on Ms

# We now define the cost function, which includes the constraint on the maximum sensitivity function
function cost(params)
    kp,ki,kd = params
    C = pid(kp=kp,ki=ki,kd=kd)
    G = feedback(P*C) # Closed-loop system
    S = 1/(1 + P*C)   # Sensitivity function
    local Gd
    try
        Gd = c2d(G,0.1) # Discretize the system. This might fail for some parameters, so we catch these cases and return a high value
    catch
        return 1000
    end
    y  = step(Gd,15)[1][:] # This is the time-domain simulation
    Ms = maximum(bode(S, w)[1]) # Maximum of the sensitivity function
    q  = quantile(Ms, 0.9)
    performance = mean(abs, 1 .- y)   # This is our performance measure
    robustness = (q > Msc ? 10000(q-Msc) : 0) # This is our robustness constraint
    variance = 10std(performance)     # This is the price we pay for high variance in the performance
    mean(performance) + robustness + variance
end

# We are now ready to test the cost function. This will take very long time to compile the first time it's called since we use StaticParticles (~60s on my machine), but should be very fast after that (~6ms)
@time cost(params)
#
res = optimize(cost, params, NelderMead(), Optim.Options(iterations=200, show_trace=false));
# We can now perform the same computations as above to visualize the found controller
fig = plot(layout=2)
for params = (params, res.minimizer)
    kp,ki,kd = params
    C  = pid(kp =kp,ki =ki,kd =kd)
    G  = feedback(P*C)
    S  = 1/(1 + P*C)
    Gd = c2d(G,0.1)
    y,t,_ = step(Gd,15)
    y     = y[:]
    mag   = bode(S, w)[1][:]
    plot!(t,y, title="Time response", subplot=1, legend=false)
    plot!(w, mag, title="Sensitivity function", xscale=:log10, yscale=:log10, subplot=2, legend=false)
end
hline!([Msc], l=(:black, :dash), subplot=2)
fig

# Other things that could potentially be relevant is adding a probabilistic constraint on the time-domain output, such as the probability of having the step response go above 1.5 must be < 0.05 etc.
