module MonteCarloMeasurements
using LinearAlgebra, Statistics, Random, StaticArrays, RecipesBase, GenericLinearAlgebra, MacroTools
using Distributed: pmap
import Base: add_sum

using Distributions, StatsBase, Requires


const DEFAULT_NUM_PARTICLES = 10000
const DEFAULT_STATIC_NUM_PARTICLES = 100

"""
The function used to reduce particles to a number for comparison. Defaults to `mean`. Change using `unsafe_comparisons`.
"""
const COMPARISON_FUNCTION = Ref{Function}(mean)
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

export ±, ∓, .., AbstractParticles,Particles,StaticParticles, MvParticles, sigmapoints, transform_moments, ≲,≳, systematic_sample, ess, outer_product, meanstd, meanvar, register_primitive, register_primitive_multi, register_primitive_single, ℝⁿ2ℝⁿ_function, ℝⁿ2ℂⁿ_function, ℂ2ℂ_function, ℂ2ℂ_function!, resample!, bootstrap, sqrt!, exp!, sin!, cos!, wasserstein
# Plot exports
export errorbarplot, mcplot, ribbonplot

# Statistics reexport
export mean, std, cov, var, quantile, median
# Distributions reexport
export Normal, MvNormal, Cauchy, Beta, Exponential, Gamma, Laplace, Uniform, fit, logpdf

export unsafe_comparisons, @unsafe, set_comparison_function

export bymap, bypmap, @bymap, @bypmap, @prob, Workspace, with_workspace, has_particles, mean_object, change_representation

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
include("particle_distributions.jl")

# This is defined here so that @bymap is loaded
LinearAlgebra.norm2(p::AbstractVector{<:AbstractParticles}) = bymap(LinearAlgebra.norm2,p)
Base.:\(x::AbstractVecOrMat{<:AbstractParticles}, y::AbstractVecOrMat{<:AbstractParticles}) = bymap(\, x, y)

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")
    @require SLEEFPirates="476501e8-09a2-5ece-8869-fb82de89a1fa" include("sleefpirates.jl")
    @require Measurements="eff96d63-e80a-5855-80a2-b1b0885c5ab7" include("measurements.jl")
    @require Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d" include("unitful.jl")
end

end
