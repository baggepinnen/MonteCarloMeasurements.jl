module MonteCarloMeasurements

const DEFAUL_NUM_PARTICLES = 500
const DEFAUL_STATIC_NUM_PARTICLES = 100

export Particles,StaticParticles, ≲,≳, systematic_sample, outer_product, meanstd, meanvar, ℝⁿ2ℝⁿ_function, ℂ2ℂ_function
export mean, std, cov, var, quantile, median
export errorbarplot, mcplot, ribbonplot

import Base: add_sum


using LinearAlgebra, Statistics, Random, StaticArrays, Reexport, RecipesBase, GenericLinearAlgebra
using Lazy: @forward

@reexport using Distributions

include("sampling.jl")
include("particles.jl")
include("diff.jl")
include("plotting.jl")

end


# TODO: Mutation, test on ControlSystems
