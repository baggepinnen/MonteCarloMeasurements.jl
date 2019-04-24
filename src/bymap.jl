import Base.Cartesian.@ntuple

const SomeKindOfParticles = Union{<:MonteCarloMeasurements.AbstractParticles, MonteCarloMeasurements.MvParticles}

nparticles(p) = length(p)
nparticles(p::MvParticles) = length(eltype(p))
nparticles(p::Type{<:MvParticles}) = length(eltype(p))

vecindex(p,i) = getindex(p,i)
vecindex(p::MvParticles,i) = getindex.(p,i)

function indexof_particles(args)
    inds = findall([a <: SomeKindOfParticles for a in args])
    inds === nothing && throw(ArgumentError("At least one argument should be <: AbstractParticles"))
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

@generated function arggetter(i,args...)
    nargs = length(args)
    inds = indexof_particles(args)
    quote
        @ntuple $nargs j->j ∈ $inds ? vecindex(args[j],i) : args[j] # Must interpolate in vars that were created outside of the quote, but not arguments to the generated function
    end
end


macro bymap(ex)
    @capture(ex, f_(args__)) || error("expected a function call")
    quote
        N = Ngetter($(esc.(args)...))
        individuals = map(1:N) do i
            argsi = arggetter(i,$(esc.(args)...))
            $(esc(f))(argsi...)
        end
        if ndims(individuals[1]) == 0
            return Particles(individuals)
        else
            return Particles(copy(reduce(hcat,individuals)'))
        end
    end
end

macro bypmap(ex)
    @capture(ex, f_(args__)) || error("expected a function call")
    quote
        N = Ngetter($(esc.(args)...))
        individuals = pmap(1:N) do i
            argsi = arggetter(i,$(esc.(args)...))
            $(esc(f))(argsi...)
        end
        if ndims(individuals[1]) == 0
            return Particles(individuals)
        else
            return Particles(copy(reduce(hcat,individuals)'))
        end
    end
end
