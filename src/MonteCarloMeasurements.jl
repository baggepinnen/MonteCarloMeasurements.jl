"""
This package facilitates working with probability distributions by means of Monte-Carlo methods, in a way that allows for propagation of probability distributions through functions. This is useful for, e.g.,  nonlinear [uncertainty propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty). A variable or parameter might be associated with uncertainty if it is measured or otherwise estimated from data. We provide two core types to represent probability distributions: `Particles` and `StaticParticles`, both `<: Real`. (The name "Particles" comes from the [particle-filtering](https://en.wikipedia.org/wiki/Particle_filter) literature.) These types all form a Monte-Carlo approximation of the distribution of a floating point number, i.e., the distribution is represented by samples/particles. **Correlated quantities** are handled as well, see [multivariate particles](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/#Multivariate-particles-1) below.

A number of type `Particles` behaves just as any other `Number` while partaking in calculations. After a calculation, an approximation to the **complete distribution** of the output is captured and represented by the output particles. `mean`, `std` etc. can be extracted from the particles using the corresponding functions. `Particles` also interact with [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), so that you can call, e.g., `Normal(p)` and get back a `Normal` type from distributions or `fit(Gamma, p)` to get a `Gamma`distribution. Particles can also be iterated, asked for `maximum/minimum`, `quantile` etc. If particles are plotted with `plot(p)`, a histogram is displayed. This requires Plots.jl. A kernel-density estimate can be obtained by `density(p)` is StatsPlots.jl is loaded.

## Quick start
```julia
julia> using MonteCarloMeasurements, Plots

julia> a = π ± 0.1 # Construct Gaussian uncertain parameters using ± (\\pm)
Particles{Float64,2000}
 3.14159 ± 0.1

julia> b = 2 ∓ 0.1 # ∓ (\\mp) creates StaticParticles (with StaticArrays)
StaticParticles{Float64,100}
 2.0 ± 0.0999

julia> std(a)      # Ask about statistical properties
0.09999231528930486

julia> sin(a)      # Use them like any real number
Particles{Float64,2000}
 1.2168e-16 ± 0.0995

julia> plot(a)     # Plot them

julia> b = sin.(1:0.1:5) .± 0.1; # Create multivariate uncertain numbers

julia> plot(b)                   # Vectors of particles can be plotted

julia> using Distributions

julia> c = Particles(500, Poisson(3.)) # Create uncertain numbers distributed according to a given distribution
Particles{Int64,500}
 2.882 ± 1.7
```

For further help, see the [documentation](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable), the [examples folder](https://github.com/baggepinnen/MonteCarloMeasurements.jl/tree/master/examples) or the [arXiv paper](https://arxiv.org/abs/2001.07625).
"""
module MonteCarloMeasurements
using LinearAlgebra, Statistics, Random, StaticArrays, RecipesBase, MacroTools, SLEEFPirates
using Distributed: pmap
import Base: add_sum

using Distributions, StatsBase, Requires


const DEFAULT_NUM_PARTICLES = 2000
const DEFAULT_STATIC_NUM_PARTICLES = 100

function pmean end

"""
The function used to reduce particles to a number for comparison. Defaults to `mean`. Change using `unsafe_comparisons`.
"""
const COMPARISON_FUNCTION = Ref{Function}(pmean)
const COMPARISON_MODE = Ref(:safe)

"""
    unsafe_comparisons(onoff=true; verbose=true)
Toggle the use of a comparison function without warning. By default `mean` is used to reduce particles to a floating point number for comparisons. This function can be changed, example: `set_comparison_function(median)`

    unsafe_comparisons(mode=:reduction; verbose=true)
One can also specify a comparison mode, `mode` can take the values `:safe, :montecarlo, :reduction`. `:safe` is the same as calling `unsafe_comparisons(false)` and `:reduction` corresponds to `true`.
If
"""
function unsafe_comparisons(mode=true; verbose=true)
    mode == false && (mode = :safe)
    mode == true && (mode = :reduction)
    COMPARISON_MODE[] = mode
    if mode != :safe && verbose
        if mode === :reduction
            @info "Unsafe comparisons using the function `$(COMPARISON_FUNCTION[])` has been enabled globally. Use `@unsafe` to enable in a local expression only or `unsafe_comparisons(false)` to turn off unsafe comparisons"
        elseif mode === :montecarlo
            @info "Comparisons using the monte carlo has been enabled globally. Call `unsafe_comparisons(false)` to turn off unsafe comparisons"
        end
    end
    mode ∉ (:safe, :montecarlo, :reduction) && error("Got unsupported comparison model")
end
"""
    set_comparison_function(f)

Change the Function used to reduce particles to a number for comparison operators
Toggle the use of a comparison Function without warning using the Function `unsafe_comparisons`.
"""
function set_comparison_function(f)
    if f in (mean, median, maximum, minimum)
        @warn "This comparison function ($(f)) is probably not the right choice, consider if you want the particle version (p$(f)) instead."
    end
    COMPARISON_FUNCTION[] = f
end

"""
    @unsafe expression
Activates unsafe comparisons for the provided expression only. The expression is surrounded by a try/catch block to robustly restore unsafe comparisons in case of exception.
"""
macro unsafe(ex)
    ex2 = if @capture(ex, assigned_vars__ = y_)
        if length(assigned_vars) == 1
            esc(assigned_vars[1])
        else
            esc.(assigned_vars[1].args)
        end
    else
        :(res)
    end
    quote
        previous_state = COMPARISON_MODE[]
        unsafe_comparisons(true, verbose=false)
        local res
        try
            res = ($(esc(ex)))
        finally
            unsafe_comparisons(previous_state, verbose=false)
        end
        $ex2 = res
    end
end

export ±, ∓, .., ⊠, ⊞, AbstractParticles,Particles,StaticParticles, MvParticles, sigmapoints, transform_moments, ≲,≳, systematic_sample, ess, outer_product, meanstd, meanvar, register_primitive, register_primitive_multi, register_primitive_single, ℝⁿ2ℝⁿ_function, ℝⁿ2ℂⁿ_function, ℂ2ℂ_function, ℂ2ℂ_function!, resample!, bootstrap, sqrt!, exp!, sin!, cos!, wasserstein, with_nominal, nominal, nparticles, particleeltype
# Plot exports
export errorbarplot, mcplot, ribbonplot

# Statistics reexport
export mean, std, cov, var, quantile, median
export pmean, pstd, pcov, pcor, pvar, pquantile, pmedian, pmiddle, piterate, pextrema, pminimum, pmaximum
# Distributions reexport
export Normal, MvNormal, Cauchy, Beta, Exponential, Gamma, Laplace, Uniform, fit, logpdf

export unsafe_comparisons, @unsafe, set_comparison_function

export bymap, bypmap, @bymap, @bypmap, @prob, Workspace, with_workspace, has_particles, mean_object

include("types.jl")
include("register_primitive.jl")
include("sampling.jl")
include("particles.jl")
include("distances.jl")
include("complex.jl")
include("sigmapoints.jl")
include("resampling.jl")
include("bymap.jl")
include("deconstruct.jl")
include("diff.jl")
include("plotting.jl")
include("optimize.jl")
include("sleefpirates.jl")
include("nominal.jl")

# This is defined here so that @bymap is loaded
LinearAlgebra.norm2(p::AbstractArray{<:AbstractParticles}) = bymap(LinearAlgebra.norm2,p)
Base.:\(x::AbstractVecOrMat{<:AbstractParticles}, y::AbstractVecOrMat{<:AbstractParticles}) = bymap(\, x, y)
Base.:\(x::Diagonal{<:AbstractParticles}, y::Vector{<:AbstractParticles}) = bymap(\, x, y) # required for ambiguity

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")
    @require Measurements="eff96d63-e80a-5855-80a2-b1b0885c5ab7" include("measurements.jl")
    @require Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d" include("unitful.jl")
end

end
