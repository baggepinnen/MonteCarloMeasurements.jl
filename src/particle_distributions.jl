struct ParticleDistribution{D,N,P}
    d::Vector{D}
    constructor::P
end


"""
    ParticleDistribution(constructor::Type{<:Distribution}, p...)

A `ParticleDistribution` represents a hierarchical distribution where the parameters of the distribution are `Particles`. The internal representation is as a `Vector{Distribution{FloatType}}` for efficient drawing of random numbers etc. But construction and printing is done as if it was a type `Distribution{Particles}`.

# Example
```julia
julia> pd = ParticleDistribution(Normal, 1±0.1, 1±0.1)
ParticleNormal{Float64}(
 μ: 1.0 ± 0.1
 σ: 1.0 ± 0.1
)

julia> rand(pd)
Part10000(1.012 ± 1.01)
"""
function ParticleDistribution(constructor::Type{<:Distribution}, p...)
    N = nparticles(p[1])
    dists = [constructor(getindex.(p, i)...) for i in 1:N]
    ParticleDistribution{eltype(dists), N, typeof(constructor)}(dists, constructor)
end

Base.length(d::ParticleDistribution) = length(d.d[1])
Base.eltype(d::ParticleDistribution{D,N}) where {D,N} = Particles{eltype(D),N}

Particles(a::BitArray) = Particles(Vector(a))

function Base.rand(rng::AbstractRNG, d::ParticleDistribution{D,N}) where {D,N}
    eltype(d)(rand.(rng, d.d))
end

Base.rand(d::ParticleDistribution) = rand(Random.GLOBAL_RNG, d)

function Base.show(io::IO, d::ParticleDistribution{D}) where D
    fields = map(fieldnames(D)) do fn
        getfield.(d.d, fn)
    end
    println(io, "Particle", D, "(")
    for (i,fn) in enumerate(fieldnames(D))
        println(io, " ", string(fn), ": ", Particles(fields[i]))
    end
    print(io, ")")
end

Base.getindex(d::ParticleDistribution, i...) = getindex(d.d, i...)


function Distributions.logpdf(pd::ParticleDistribution{D,N}, x) where {D,N}
    T = float(eltype(D))
    Particles{T,N}(logpdf.(pd.d, x))
end
