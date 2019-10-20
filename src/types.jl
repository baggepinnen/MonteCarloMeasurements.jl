const ConcreteFloat = Union{Float64,Float32,Float16,BigFloat}
const ConcreteInt = Union{Bool,Int8,Int16,Int32,Int64,Int128,BigInt}

abstract type AbstractParticles{T,N} <: Real end
"""
    struct Particles{T, N} <: AbstractParticles{T, N}

This type represents uncertainty using a cloud of particles.
# Constructors:
- `Particles()`
- `Particles(N::Integer)`
- `Particles(d::Distribution)`
- `Particles(N::Integer, d::Distribution)`
- `Particles(N::Integer, d::Distribution; permute, systematic)`
- `Particles(v::Array{T,1} where T)`
- `Particles(m::Array{T,2} where T)`
"""
struct Particles{T,N} <: AbstractParticles{T,N}
    particles::Vector{T}
end

"""
    struct StaticParticles{T, N} <: AbstractParticles{T, N}

See `?Particles` for help. The difference between `StaticParticles` and `Particles` is that the `StaticParticles` store particles in a static vecetor. This makes runtimes much shorter, but compile times longer. See the documentation for some benchmarks.
"""
struct StaticParticles{T,N} <: AbstractParticles{T,N}
    particles::SArray{Tuple{N}, T, 1, N}
end

"""
Particles with weights.
To weight the particles `p`, modify the field `p.logweights`. You can resample the particles using `resample!(p)`, where each particles is resampled with a probability proportional to its weight.
"""
struct WeightedParticles{T,N} <: AbstractParticles{T,N}
    particles::Vector{T}
    # weights::Vector{T}
    logweights::Vector{T}
end
function WeightedParticles{T,N}(v::AbstractVector) where {T,N}
    # weights = fill(1/N, N)
    logweights = fill(-log(N), N)
    WeightedParticles{T,N}(v,logweights)
end



const MvParticles = Vector{<:AbstractParticles} # This can not be AbstractVector since it causes some methods below to be less specific than desired
const MvWParticles = Vector{<:WeightedParticles}
