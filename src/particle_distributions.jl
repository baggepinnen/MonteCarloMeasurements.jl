struct ParticleDistribution{D,P}
    d::D
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
2.2224400199728356
"""
function ParticleDistribution(constructor::Type{<:Distribution}, p...)
    dists = [constructor(getindex.(p, i)...) for i in 1:nparticles(p[1])]
    ParticleDistribution(dists, constructor)
end

Base.length(d::ParticleDistribution) = length(d.d[1])
Base.eltype(d::ParticleDistribution) = eltype(eltype(d.d))

function Base.rand(rng::AbstractRNG, d::ParticleDistribution)
    ind = rand(rng, 1:length(d.d))
    rand(rng, d.d[ind])
end

Base.rand(d::ParticleDistribution) = rand(Random.GLOBAL_RNG, d)

function Base.show(io::IO, d::ParticleDistribution)
    T = eltype(d.d)
    fields = map(fieldnames(T)) do fn
        getfield.(d.d, fn)
    end
    println(io, "Particle", T, "(")
    for (i,fn) in enumerate(fieldnames(T))
        println(io, " ", string(fn), ": ", Particles(fields[i]))
    end
    print(io, ")")
end

Base.getindex(d::ParticleDistribution, i...) = getindex(d.d, i...)
