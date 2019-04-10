module MonteCarloMeasurements

export Particles,StaticParticles, ≲, systematic_sample
export mean, std, cov, var, quantile, median
export errorbarplot, mcplot, ribbonplot


using LinearAlgebra, Statistics, Random, StaticArrays, Reexport, RecipesBase, GenericLinearAlgebra
using Lazy: @forward

@reexport using Distributions

include("sampling.jl")
include("particles.jl")
include("diff.jl")

end


# TODO: Mutation, test on ControlSystems
# TODO: InplaceParticles: maintains an output workspace
# struct InplaceParticles
#     particles::Vector
#     output::ImmutableVector
# end
# This output should ideally be some kind of immutable vector since it could be unsafe to mutate it. Unclear how the map!(f, output, particles) would be done if output is immutable though.
