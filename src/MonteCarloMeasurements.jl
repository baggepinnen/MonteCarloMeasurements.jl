module MonteCarloMeasurements

export Particles,StaticParticles, ≲


using LinearAlgebra, Statistics, Random, StaticArrays, Reexport, RecipesBase
using Lazy: @forward

@reexport using Distributions

include("particles.jl")

# InteractiveUtils, which, names, namesof
# TODO: Broadcast, Mutation, Code generation or define for all base functions, test on ControlSystems
# module Operators
# end
end # module
