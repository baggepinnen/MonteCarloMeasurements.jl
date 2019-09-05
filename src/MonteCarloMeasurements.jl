module MonteCarloMeasurements
using LinearAlgebra, Statistics, Random, StaticArrays, RecipesBase, GenericLinearAlgebra, MacroTools
using Distributed: pmap
import StatsBase: ProbabilityWeights
using Lazy: @forward
import Base: add_sum

using Distributions, StatsBase, Requires


const DEFAUL_NUM_PARTICLES = 500
const DEFAUL_STATIC_NUM_PARTICLES = 100

const COMPARISON_FUNCTION = Ref{Function}(mean)
const USE_UNSAFE_COMPARIONS = Ref(false)

"""
    unsafe_comparisons(onoff=true; verbose=true)
Toggle the use of a comparison function without warning. By default `mean` is used to reduce particles to a floating point number for comparisons. This function can be changed, example: `set_comparison_function(median)`
"""
function unsafe_comparisons(onoff=true; verbose=true)
    USE_UNSAFE_COMPARIONS[] = onoff
    if onoff && verbose
        @info "Unsafe comparisons using the function `$(COMPARISON_FUNCTION[])` has been enabled globally. Use `@unsafe` to enable in a local expression only or `unsafe_comparisons(false)` to turn off unsafe comparisons"
    end
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
        previous_state = USE_UNSAFE_COMPARIONS[]
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

export AbstractParticles,Particles,StaticParticles, WeightedParticles, sigmapoints, transform_moments, ≲,≳, systematic_sample, outer_product, meanstd, meanvar, register_primitive, register_primitive_multi, register_primitive_single, ℝⁿ2ℝⁿ_function, ℂ2ℂ_function, ℂ2ℂ_function!, resample!, sqrt!, exp!, sin!, cos!
# Plot exports
export errorbarplot, mcplot, ribbonplot

# Statistics reexport
export mean, std, cov, var, quantile, median
# Distributions reexport
export Normal, MvNormal, Cauchy, Beta, Exponential, Gamma, Laplace, Uniform, fit, logpdf

export unsafe_comparisons, @unsafe, set_comparison_function

export @bymap, @bypmap, Workspace, with_workspace, has_particles, mean_object

include("types.jl")
include("register_primitive.jl")
include("sampling.jl")
include("particles.jl")
include("complex.jl")
include("sigmapoints.jl")
include("resampling.jl")
include("bymap.jl")
include("deconstruct.jl")
include("diff.jl")
include("plotting.jl")
include("optimize.jl")

# This is defined here so that @bymap is loaded
Base.log(p::Matrix{<:AbstractParticles}) = @bymap log(p) # Matrix more specific than StridedMatrix used in Base.log

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")
end

end


# TODO: ifelse?
