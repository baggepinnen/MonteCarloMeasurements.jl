import Base.Cartesian.@ntuple

nparticles(p) = length(p)
nparticles(p::Type) = 1
nparticles(p::Type{<:AbstractParticles{T,N}}) where {T,N} = N
nparticles(p::AbstractParticles{T,N}) where {T,N} = N
nparticles(p::ParticleArray) = nparticles(eltype(p))
nparticles(p::Type{<:ParticleArray}) = nparticles(eltype(p))

particletype(p::AbstractParticles) = typeof(p)
particletype(::Type{P}) where P <: AbstractParticles = P
particletype(p::AbstractArray{<:AbstractParticles}) = eltype(p)

particleeltype(::AbstractParticles{T,N}) where {T,N} = T
particleeltype(::AbstractArray{<:AbstractParticles{T,N}}) where {T,N} = T

"""
vecindex(p::Number,i) = p
vecindex(p,i) = getindex(p,i)
vecindex(p::AbstractParticles,i) = getindex(p.particles,i)
vecindex(p::ParticleArray,i) = vecindex.(p,i)
vecindex(p::NamedTuple,i) = (; Pair.(keys(p), ntuple(j->arggetter(i,p[j]), fieldcount(typeof(p))))...)
"""
vecindex(p::Number,i) = p
vecindex(p,i) = getindex(p,i)
vecindex(p::AbstractParticles,i) = getindex(p.particles,i)
vecindex(p::ParticleArray,i) = vecindex.(p,i)
vecindex(p::NamedTuple,i) = (; Pair.(keys(p), ntuple(j->arggetter(i,p[j]), fieldcount(typeof(p))))...)

function indexof_particles(args)
    inds = findall(a-> a <: SomeKindOfParticles, args)
    inds === nothing && throw(ArgumentError("At least one argument should be <: AbstractParticles. If particles appear nested as fields inside an argument, see `with_workspace` and `Workspace`"))
    all(nparticles(a) == nparticles(args[inds[1]]) for a in args[inds]) || throw(ArgumentError("All p::Particles must have the same number of particles."))
    (inds...,)
    # TODO: test all same number of particles
end


function arggetter(i,a::Union{SomeKindOfParticles, NamedTuple})
    vecindex(a,i)
end

arggetter(i,a) = a


"""
    @bymap f(p, args...)

Call `f` with particles or vectors of particles by using `map`. This can be utilized if registering `f` using [`register_primitive`](@ref) fails. See also [`Workspace`](@ref) if `bymap` fails.
"""
macro bymap(ex)
    @capture(ex, f_(args__)) || error("expected a function call")
    quote
        bymap($(esc(f)),$(esc.(args)...))
    end
end

"""
    bymap(f, args...)

Uncertainty propagation using the `map` function.

Call `f` with particles or vectors of particles by using `map`. This can be utilized if registering `f` using [`register_primitive`](@ref) fails. See also [`Workspace`](@ref) if `bymap` fails.
"""
function bymap(f::F, args...) where F
    inds = indexof_particles(typeof.(args))
    T,N,PT = particletypetuple(args[first(inds)])
    individuals = map(1:N) do i
        argsi = ntuple(j->arggetter(i,args[j]), length(args))
        f(argsi...)
    end
    PTNT = PT{eltype(eltype(individuals)),N}
    if (eltype(individuals) <: AbstractArray{TT,0} where TT) || eltype(individuals) <: Number
        PTNT(individuals)
    elseif eltype(individuals) <: AbstractArray{TT,1} where TT
        PTNT(copy(reduce(hcat,individuals)'))
    elseif eltype(individuals) <: AbstractArray{TT,2} where TT
        # @show PT{eltype(individuals),N}
        reshape(PTNT(copy(reduce(hcat,vec.(individuals))')), size(individuals[1],1),size(individuals[1],2))::Matrix{PTNT}
    else
        error("Output with dimension >2 is currently not supported by `bymap`. Consider if `ℝⁿ2ℝⁿ_function($(f), $(args...))` works for your use case.")
    end
end

"""
Distributed uncertainty propagation using the `pmap` function. See [`bymap`](@ref) for more details.
"""
function bypmap(f::F, args...) where F
    inds = indexof_particles(typeof.(args))
    T,N,PT = particletypetuple(args[first(inds)])
    individuals = map(1:N) do i
        argsi = ntuple(j->arggetter(i,args[j]), length(args))
        f(argsi...)
    end
    PTNT = PT{eltype(eltype(individuals)),N}
    if (eltype(individuals) <: AbstractArray{TT,0} where TT) || eltype(individuals) <: Number
        PTNT(individuals)
    elseif eltype(individuals) <: AbstractArray{TT,1} where TT
        PTNT(copy(reduce(hcat,individuals)'))
    elseif eltype(individuals) <: AbstractArray{TT,2} where TT
        # @show PT{eltype(individuals),N}
        reshape(PTNT(copy(reduce(hcat,vec.(individuals))')), size(individuals[1],1),size(individuals[1],2))::Matrix{PTNT}
    else
        error("Output with dimension >2 is currently not supported by `bymap`. Consider if `ℝⁿ2ℝⁿ_function($(f), $(args...))` works for your use case.")
    end
end

"""
    @bypmap f(p, args...)

Call `f` with particles or vectors of particles by using parallel `pmap`. This can be utilized if registering `f` using [`register_primitive`](@ref) fails. See also [`Workspace`](@ref) if `bymap` fails.
"""
macro bypmap(ex)
    @capture(ex, f_(args__)) || error("expected a function call")
    quote
        bypmap($(esc(f)),$(esc.(args)...))
    end
end


"""
    @prob a < b

Calculate the probability that an event on any of the forms `a < b, a > b, a <= b, a >= b` occurs, where `a` and/or `b` are of type `AbstractParticles`.
"""
macro prob(ex)
    ex.head == :call && ex.args[1] ∈ (:<,:>,:<=,:>=) || error("Expected an expression on any of the forms `a < b, a > b, a <= b, a >= b`")
    op = ex.args[1]
    a  = ex.args[2]
    b  = ex.args[3]
    quote
        mean($op.(MonteCarloMeasurements.maybe_particles($(esc(a))), MonteCarloMeasurements.maybe_particles($(esc(b)))))
    end
end
