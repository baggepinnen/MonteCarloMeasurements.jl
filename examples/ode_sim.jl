using MonteCarloMeasurements, Measurements, Plots, BenchmarkTools, OrdinaryDiffEq, PrettyTables, ChangePrecision, LinearAlgebra, Savefig
# pgfplots()
default(size=(600,400))
function sim((±)::F, tspan, plotfun=plot!, args...; kwargs...) where F
    @changeprecision Float32 begin
        g = 9.79 ± 0.02; # Gravitational constant
        L = 1.00 ± 0.01; # Length of the pendulum
        u₀ = [0.0 ± 0.0, π / 3.0 ± 0.02] # Initial speed and initial angle
        # @show typeof(u₀)
        gL = g/L
        #Define the problem
        function simplependulum(du,u,p,t)
            θ  = u[1]
            dθ = u[2]
            du[1] = dθ
            du[2] = -gL * sin(θ)
        end

        prob = ODEProblem(simplependulum, u₀, tspan)
        sol = solve(prob, Tsit5(), reltol = 1e-6)

        plotfun(sol.t, getindex.(sol.u, 2), args...; kwargs...)
    end
end

## Special function needed to construct the parameters as sigmapoints, as they all have to be constructed in one single call
function sigmasim(tspan, plotfun=plot!, args...; kwargs...) where F
    @changeprecision Float32 begin
        g,L,u02 = StaticParticles(sigmapoints([9.79, 1.0, pi/3.0], diagm([0.02, 0.01, 0.02].^2)))
        u₀ = [0, u02] # Initial speed and initial angle
        # @show typeof(u₀)
        gL = g/L
        #Define the problem
        function simplependulum(du,u,p,t)
            θ  = u[1]
            dθ = u[2]
            du[1] = dθ
            du[2] = -gL * sin(θ)
        end

        prob = ODEProblem(simplependulum, u₀, tspan)
        sol = solve(prob, Tsit5(), reltol = 1e-6)

        plotfun(sol.t, getindex.(sol.u, 2), args...; kwargs...)
    end
end

##
tspan = (0.0f0, 0.5f0)
plot()
sim(Measurements.:±, tspan, label = "Linear", xlims=(tspan[2]-2,tspan[2]), color=Savefig.color_palette[4])
sim(MonteCarloMeasurements.:±, tspan, errorbarplot!, 0.8413, label = "MCM", xlims=(tspan[2]-0.5,tspan[2]), l=(:dot,), color=Savefig.color_palette[2], xlabel="Time [s]", ylabel="\$\\theta\$")

# savefig("/home/fredrikb/mcm_paper/figs/0-2.pdf")
##
tspan = (0.0f0, 200)
plot()
sim(Measurements.:±, tspan, label = "Linear", xlims=(tspan[2]-5,tspan[2]), color=Savefig.color_palette[4])
sim(MonteCarloMeasurements.:±, tspan, label = "Monte Carlo", xlims=(tspan[2]-5,tspan[2]), l=(:dot,), color=Savefig.color_palette[2], xlabel="Time [s]", ylabel="\$\\theta\$")

##
# We now integrated over 200 seconds and look at the last 5 seconds. This result maybe looks a bit confusing, the linear uncertainty propagation is very sure about the amplitude at certain points but not at others, whereas the Monte-Carlo approach is completely unsure. Furthermore, the linear approach thinks that the amplitude at some points is actually much higher than the starting amplitude, implying that energy somehow has been added to the system! The picture might become a bit more clear by plotting the individual trajectories of the particles

tspan = (0.0f0, 200)
plot()
sim(Measurements.:±, tspan, label = "Linear", xlims=(tspan[2]-5,tspan[2]), l=(5,), color=Savefig.color_palette[4])
sim(MonteCarloMeasurements.:∓, tspan, mcplot!, xlims=(tspan[2]-5,tspan[2]), l=(Savefig.color_palette[2],0.3), xlabel="Time [s]", ylabel="\$\\theta\$", label="MCM", primary=onlyone(true))
sigmasim(tspan, mcplot!, xlims=(tspan[2]-5,tspan[2]), l=(Savefig.color_palette[3],0.9,2), xlabel="Time [s]", ylabel="\$\\theta\$", label="MCM \\Sigma", primary=onlyone(true))
# savefig("/home/fredrikb/mcm_paper/figs/mcplot.pdf")

# It now becomes clear that each trajectory has a constant amplitude (although individual trajectories amplitudes vary slightly due to the uncertainty in the initial angle), but the phase is all mixed up due to the slightly different frequencies!

# These problems grow with increasing uncertainty and increasing integration time. In fact, the uncertainty reported by Measurements.jl goes to infinity as the integration time does the same.

# Of course, the added accuracy from using MonteCarloMeasurements does not come for free, as it costs some additional computation. We have the following timings for integrating the above system 100 seconds using three different uncertainty representations

##
function naive_mc(tspan)
    for i = 1:100
        sim(certain, tspan, (args...;kwargs...)->nothing)
    end
end

tspan = (0.0f0, 100f0)
certain = (x,y)->x+y*randn()
table = Matrix{Any}(undef,3,6)
t1 = @benchmark sim($certain, $tspan, (args...;kwargs...)->nothing)
t2 = @benchmark sim($Measurements.:±, $tspan, (args...;kwargs...)->nothing) samples=500
t3 = @benchmark sim($MonteCarloMeasurements.:∓, $tspan, (args...;kwargs...)->nothing) samples=500
t4 = @benchmark sigmasim($tspan, (args...;kwargs...)->nothing) samples=500
t5 = @benchmark naive_mc($tspan) samples=500


# table[1,1] = ""
table[1,1] = "Time [ms]"
table[2,1] = "Memory [MiB]"
table[3,1] = "k Allocations"

for (i,t) in enumerate((t1,t2,t3,t4,t5))
    table[1,i+1] = time(t)/1000_000
    table[2,i+1] = memory(t)/1000_000
    table[3,i+1] = allocs(t)/1000
end


pretty_table(table, ["" "Float32" "Linear" "MCM" "MCM \\Sigma" "Naive MC"], backend=:latex, formatter=ft_printf("%5.1f"))
