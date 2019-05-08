module MonteCarloMeasurements
using LinearAlgebra, Statistics, Random, StaticArrays, RecipesBase, GenericLinearAlgebra, MacroTools
using Distributed: pmap
import StatsBase: ProbabilityWeights
using Lazy: @forward
import Base: add_sum

using Distributions, StatsBase


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
        @info "Unsafe comparisons using the function $(COMPARISON_FUNCTION[]) has been enabled globally. Use `@unsafe_comparisons` to enable in a local expression only or `unsafe_comparisons(false)` to turn off unsafe comparisons"
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
# TODO: have to figure out if new bindings are created in ex and if so, declare them local outside the try block
using Base.Cartesian: @nexprs
macro unsafe_comparisons(ex)
    @capture(ex, assigned_vars__ = y_)
    n = length(assigned_vars)
    quote
        previous_state = USE_UNSAFE_COMPARIONS[]
        unsafe_comparisons(true, verbose=false)
        @nexprs $n j->(local $(esc(assigned_vars[j])))
        local res
        try
            res = ($(esc(ex)))
        finally
            unsafe_comparisons(previous_state, verbose=false)
        end
        res
    end
end

export AbstractParticles,Particles,StaticParticles, WeightedParticles, sigmapoints, transform_moments, ≲,≳, systematic_sample, outer_product, meanstd, meanvar, register_primitive, register_primitive_multi, register_primitive_single, ℝⁿ2ℝⁿ_function, ℂ2ℂ_function, resample!, @bymap, @bypmap
# Plot exports
export errorbarplot, mcplot, ribbonplot

# Statistics reexport
export mean, std, cov, var, quantile, median
# Distributions reexport
export Normal, MvNormal, Cauchy, Beta, Exponential, Gamma, Laplace, Uniform, fit, logpdf

export unsafe_comparisons, @unsafe_comparisons, set_comparison_function



include("types.jl")
include("register_primitive.jl")
include("sampling.jl")
include("particles.jl")
include("sigmapoints.jl")
include("resampling.jl")
include("bymap.jl")
include("diff.jl")
include("plotting.jl")
include("optimize.jl")

end


# TODO: ifelse?
