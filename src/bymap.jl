import Base.Cartesian.@ntuple

const ParticleArray = AbstractArray{<:AbstractParticles}
const SomeKindOfParticles = Union{<:MonteCarloMeasurements.AbstractParticles, ParticleArray}


nparticles(p) = length(p)
nparticles(p::ParticleArray) = length(eltype(p))
nparticles(p::AbstractParticles{T,N}) where {T,N} = N
nparticles(p::Type{<:ParticleArray}) = length(eltype(p))

particletype(p::AbstractParticles) = typeof(p)
particletype(::Type{P}) where P <: AbstractParticles = P
particletype(p::AbstractArray{<:AbstractParticles}) = eltype(p)

@inline vecindex(p,i) = getindex(p,i)
@inline vecindex(p::ParticleArray,i) = getindex.(p,i)
@inline vecindex(p::AbstractParticles{T,N},i::AbstractVector) where {T,N} = GenericParticles{T,length(i)}(uview(p,i))
@inline vecindex(p::NamedTuple,i) = (; Pair.(keys(p), ntuple(j->arggetter(i,p[j]), fieldcount(typeof(p))))...)



function indexof_particles(args)
    inds = findall(a-> a <: SomeKindOfParticles, [args...])
    inds === nothing && throw(ArgumentError("At least one argument should be <: AbstractParticles. If particles appear nested as fields inside an argument, see `with_workspace` and `Workspace`"))
    all(nparticles(a) == nparticles(args[inds[1]]) for a in args[inds]) || throw(ArgumentError("All p::Particles must have the same number of particles."))
    (inds...,)
    # TODO: test all same number of particles
end

@gg function argsigetter(a)
    T,N,PT = particletypetuple(a)
    :($T,$N,$PT,i->(GenericParticles(uview(a,i)),))
end

@gg function argsigetter(a1,a2)
    args = (a1,a2)
    inds = indexof_particles(args)
    T,N,PT = particletypetuple(args[first(inds)])
    nargs = length(args)
    :($T,$N,$PT,i->(arggetter(i, a1), arggetter(i, a1)))
end

@inline arggetter(i,a::Union{SomeKindOfParticles, NamedTuple}) = vecindex(a,i)

@inline arggetter(i,a) = a

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

# p = 1 ± 1
# bymap(sin, p) == sin(p)


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

Base.getindex(p::AbstractParticles, i::UnitRange) = getindex(p.particles, i)

function chunkmap(f::F, chunk_size::Int, args::Tuple) where F
    T,N,PT,argsi = argsigetter(args...)
    # @assert nextpow(2,N) == N "N must currently be a power of two for chunkmap"
    nargs = length(args)

    @assert N % chunk_size == 0 "chunk_size must be a factor of N for chunkmap"
    nsim::Int = N÷chunk_size
    # nt = Threads.nthreads()
    @assert nsim > 1
    res1 = f(argsi(1:chunk_size)...)
    chunkmap_inner(f, res1, nsim, chunk_size, argsi)

end

function chunkmap_inner(f::F, res1::T, nsim, chunk_size, argsi) where {F,T}
    # nargs = length(args)
    individuals = Vector{T}(undef, nsim)
    individuals[1] = res1
    # @show typeof(args)
    Threads.@threads for i in Base.OneTo(nsim-1)
        inds = i*chunk_size .+ (1:chunk_size)
        # argsi = @ntuple($nargs, j->arggetter(inds, args[j]))
        individuals[i+1] = f(argsi(inds)...)
    end
    p = pmerge(Particles{T,nsim}(individuals))

end

function pmerge(pi::Particles{<:AbstractParticles{T,Ni}, No}) where {T,Ni,No}
    v = Vector{T}(undef, Ni*No)
    inds = 1:Ni
    for (i,p) in enumerate(pi)
        v[(i-1)*Ni .+ inds] .= p.particles
    end
    Particles{T,Ni*No}(v)
end

pmerge(pi::AbstractArray{<:AbstractParticles}) = pmerge.(pi)
