@inline maybe_particles(x) = x
@inline maybe_particles(p::AbstractParticles) = p.particles
@inline maybe_logweights(x) = 0
@inline maybe_logweights(p::WeightedParticles) = p.logweights

function register_primitive(ff)
    register_primitive_multi(ff)
    register_primitive_single(ff)
end

function register_primitive_multi(ff)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        @eval begin
            function ($m.$f)(p::$PT{T,N},a::Real...) where {T,N}
                $PT{T,N}(($m.$f).(p.particles, maybe_particles.(a)...)) # maybe_particles introduced to handle >2 arg operators
            end
            function ($m.$f)(a::Real,p::$PT{T,N}) where {T,N}
                $PT{T,N}(map(x->($m.$f)(a,x), p.particles))
            end
            function ($m.$f)(p1::$PT{T,N},p2::$PT{T,N}) where {T,N}
                $PT{T,N}(map(($m.$f), p1.particles, p2.particles))
            end
            function ($m.$f)(p1::$PT{T,N},p2::$PT{S,N}) where {T,S,N} # Needed for particles of different float types :/
                $PT{promote_type(T,S),N}(map(($m.$f), p1.particles, p2.particles))
            end
        end
    end
end

function register_primitive_single(ff)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        @eval function ($m.$f)(p::$PT)
            $PT(map(($m.$f), p.particles))
        end
    end
end
