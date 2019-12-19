import Base.Cartesian.@ntuple

const ParticleArray = AbstractArray{<:AbstractParticles}
const SomeKindOfParticles = Union{<:MonteCarloMeasurements.AbstractParticles, ParticleArray}


nparticles(p) = length(p)
nparticles(p::ParticleArray) = length(eltype(p))
nparticles(p::Type{<:ParticleArray}) = length(eltype(p))

vecindex(p,i) = getindex(p,i)
vecindex(p::ParticleArray,i) = getindex.(p,i)
vecindex(p::NamedTuple,i) = (; Pair.(keys(p), ntuple(j->arggetter(i,p[j]), fieldcount(typeof(p))))...)

function indexof_particles(args)
    inds = findall([a <: SomeKindOfParticles for a in args])
    inds === nothing && throw(ArgumentError("At least one argument should be <: AbstractParticles. If particles appear nested as fields inside an argument, see `with_workspace` and `Workspace`"))
    all(nparticles(a) == nparticles(args[inds[1]]) for a in args[inds]) || throw(ArgumentError("All p::Particles must have the same number of particles."))
    (inds...,)
    # TODO: test all same number of particles
end

@generated function Ngetter(args...)
    nargs = length(args)
    inds = indexof_particles(args)
    N = nparticles(args[inds[1]])
    :($N)
end

function arggetter(i,a::Union{SomeKindOfParticles, NamedTuple})
    vecindex(a,i)
end

function arggetter(i,a)
    a
end

"""
    @bymap f(p, args...)

Call `f` with particles or vectors of particles by using `map`. This can be utilized if registering `f` using [`register_primitive`](@ref) fails. See also [`Workspace`](@ref) if `bymap` fails.
"""
macro bymap(ex)
    @capture(ex, f_(args__)) || error("expected a function call")
    fsym = string(f)
    nargs = length(args)
    quote
        N = Ngetter($(esc.(args)...))
        individuals = map(1:N) do i
            eargs = ($(esc.(args)...),)
            argsi = ntuple(j->arggetter(i,eargs[j]), $nargs)
            $(esc(f))(argsi...)
        end
        if ndims(individuals[1]) == 0
            Particles(individuals)
        elseif ndims(individuals[1]) == 1
            Particles(copy(reduce(hcat,individuals)'))
        elseif ndims(individuals[1]) == 2
            reshape(Particles(copy(reduce(hcat,vec.(individuals))')), size(individuals[1])...)
        else
            error("Output with dimension >2 is currently not supported by `@bymap`. Consider if `ℝⁿ2ℝⁿ_function($($fsym), $($args...))` works for your use case.")
        end
    end
end


"""
    @bypmap f(p, args...)

Call `f` with particles or vectors of particles by using parallel `pmap`. This can be utilized if registering `f` using [`register_primitive`](@ref) fails. See also [`Workspace`](@ref) if `bymap` fails.
"""
macro bypmap(ex)
    @capture(ex, f_(args__)) || error("expected a function call")
    fsym = string(f)
    nargs = length(args)
    quote
        N = Ngetter($(esc.(args)...))
        individuals = pmap(1:N) do i
            eargs = ($(esc.(args)...),)
            argsi = ntuple(j->arggetter(i,eargs[j]), $nargs)
            $(esc(f))(argsi...)
        end
        if ndims(individuals[1]) == 0
            Particles(individuals)
        elseif ndims(individuals[1]) == 1
            Particles(copy(reduce(hcat,individuals)'))
        elseif ndims(individuals[1]) == 2
            reshape(Particles(copy(reduce(hcat,vec.(individuals))')), size(individuals[1])...)
        else
            error("Output with dimension >2 is currently not supported by `@bymap`. Consider if `ℝⁿ2ℝⁿ_function($($fsym), $($args...))` works for your use case.")
        end
    end
end
