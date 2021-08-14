# In this script, we will design a PID controller by optimization. The system model has uncertain parameters, and we pay a price not only for poor performance of the closed-loop system, but also for a high variance in the performance. In addition to this, we place a constraint on the 90:th percentile of the maximum of the sensitivity function. This way, we will get a doubly robust controller as a result :) To avoid excessive amplification of measurement noise, we penalize noise amplification above a certain frequency.
# We start by defining the system and some initial controller parameters
using MonteCarloMeasurements, Optim, ControlSystems, Plots
using MonteCarloMeasurements: ∓
unsafe_comparisons(true)
p = 1 + 0.1*Particles(100)
ζ = 0.3 + 0.05*Particles(100)
ω = 1 + 0.05*Particles(100)
const P = tf([p*ω], [1, 2ζ*ω, ω^2]) |> ss
const w = exp10.(LinRange(-2,3,100))
params = log.([1,0.1,0.1])
const Msc = 1.2 # Constraint on Ms

# We now define the cost function, which includes the constraint on the maximum sensitivity function

function systems(params::AbstractVector{T}) where T
    kp,ki,kd = exp.(params)
    C = convert(StateSpace{Continuous, T}, pid(kp=kp,ki=ki,kd=kd)*tf(1, [0.05, 1])^2, balance=false)
    G = feedback(P*C) # Closed-loop system
    S = 1/(1 + P*C)   # Sensitivity function
    CS = C*S          # Noise amplification
    local Gd
    try
        Gd = c2d(G,0.1) # Discretize the system. This might fail for some parameters, so we catch these cases and return a high value
    catch
        return T(10000)
    end
    y,t,_  = step(Gd,15) .|> vec # This is the time-domain simulation
    C, G, S, CS, y, t
end

function cost(params::AbstractVector{T}) where T
    C, G, S, CS, y, t = systems(params)
    Ms = maximum(bode(S, w)[1]) # Maximum of the sensitivity function
    q  = pquantile(Ms, 0.9)
    performance = mean(abs, 1 .- y)   # This is our performance measure
    robustness = (q > Msc ? 10000(q-Msc) : zero(T)) # This is our robustness constraint
    variance = pstd(performance)     # This is the price we pay for high variance in the performance
    noise = pmean(sum(bode(CS, w[end-30:end])[1]))
    100pmean(performance) + robustness  + 10variance + 0.002noise
end

# We are now ready to test the cost function.
@time cost(params)
#
res = Optim.optimize(cost, params, NelderMead(), Optim.Options(iterations=1000, show_trace=true, show_every=20));
println("Final cost: ", res.minimum)
# We can now perform the same computations as above to visualize the found controller
fig = plot(layout=2)
for params = (params, res.minimizer)
    C, G, S, CS, y, t = systems(params)
    mag   = bode(S, w)[1][:]
    plot!(t,y[:], title="Time response", subplot=1, legend=false)
    plot!(w, mag, title="Sensitivity function", xscale=:log10, yscale=:log10, subplot=2, legend=false)
end
hline!([Msc], l=(:black, :dash), subplot=2)
display(fig)

# Other things that could potentially be relevant is adding a probabilistic constraint on the time-domain output, such as the probability of having the step response go above 1.5 must be < 0.05 etc.
