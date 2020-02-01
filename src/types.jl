const ConcreteFloat = Union{Float64,Float32,Float16,BigFloat}
const ConcreteInt = Union{Bool,Int8,Int16,Int32,Int64,Int128,BigInt}

abstract type AbstractParticles{T,N} <: Real end
"""
    struct Particles{T, N} <: AbstractParticles{T, N}

This type represents uncertainty using a cloud of particles.
# Constructors:
- `Particles()`
- `Particles(N::Integer)`
- `Particles([rng::AbstractRNG,] d::Distribution)`
- `Particles([rng::AbstractRNG,] N::Integer, d::Distribution; permute=true, systematic=true)`
- `Particles(v::Vector{T} where T)`
- `Particles(m::Matrix{T} where T)`: Creates multivariate particles (Vector{Particles})
"""
struct Particles{T,N} <: AbstractParticles{T,N}
    particles::Vector{T}
end

"""
    struct StaticParticles{T, N} <: AbstractParticles{T, N}

See `?Particles` for help. The difference between `StaticParticles` and `Particles` is that the `StaticParticles` store particles in a static vecetor. This makes runtimes much shorter, but compile times longer. See the documentation for some benchmarks. Only recommended for sample sizes of ≲ 300-400
"""
struct StaticParticles{T,N} <: AbstractParticles{T,N}
    particles::SArray{Tuple{N}, T, 1, N}
end

for PT in (:Particles, :StaticParticles)
    for D in (2,3,4,5)
        @eval function $PT{T,N}(m::AbstractArray{T,$D}) where {T,N}
            size(m, 1) == N || throw(ArgumentError("The first dimension of the matrix must be the same as the number N of particles."))
            inds = CartesianIndices(axes(m)[2:end])
            map(inds) do ind
                $PT{T,N}(@view(m[:,ind]))
            end
        end

        @eval function $PT(m::AbstractArray{T,$D}) where T
            N = size(m, 1)
            inds = CartesianIndices(axes(m)[2:end])
            map(inds) do ind
                $PT{T,N}(@view(m[:,ind]))
            end
        end
    end

    @eval begin
        $PT(v::Vector) = $PT{eltype(v),length(v)}(v)

        function $PT{T,N}(n::Real) where {T,N} # This constructor is potentially dangerous, replace with convert?
            v = fill(n,N)
            $PT{T,N}(v)
        end

        $PT{T,N}(p::$PT{T,N}) where {T,N} = p

        function $PT(rng::AbstractRNG, N::Integer=DEFAULT_NUM_PARTICLES, d::Distribution{<:Any,VS}=Normal(0,1); permute=true, systematic=VS==Continuous) where VS
            if systematic
                v = systematic_sample(rng,N,d; permute=permute)
            else
                v = rand(rng, d, N)
            end
            $PT{eltype(v),N}(v)
        end
        function $PT(N::Integer=DEFAULT_NUM_PARTICLES, d::Distribution{<:Any,VS}=Normal(0,1); kwargs...) where VS
            return $PT(Random.GLOBAL_RNG, N, d; kwargs...)
        end

        function $PT(rng::AbstractRNG, N::Integer, d::MultivariateDistribution)
            v = rand(rng,d,N)' |> copy # For cache locality
            $PT(v)
        end
        $PT(N::Integer, d::MultivariateDistribution) = $PT(Random.GLOBAL_RNG, N, d)

        nakedtypeof(p::$PT{T,N}) where {T,N} = $PT
        nakedtypeof(::Type{$PT{T,N}}) where {T,N} = $PT
    end
end

function Particles(rng::AbstractRNG, d::Distribution;kwargs...)
    Particles(rng, DEFAULT_NUM_PARTICLES, d; kwargs...)
end
Particles(d::Distribution; kwargs...) = Particles(Random.GLOBAL_RNG, d; kwargs...)

function StaticParticles(rng::AbstractRNG, d::Distribution;kwargs...)
    StaticParticles(rng, DEFAULT_STATIC_NUM_PARTICLES, d; kwargs...)
end
StaticParticles(d::Distribution;kwargs...) = StaticParticles(Random.GLOBAL_RNG, d; kwargs...)


const MvParticles = Vector{<:AbstractParticles} # This can not be AbstractVector since it causes some methods below to be less specific than desired
