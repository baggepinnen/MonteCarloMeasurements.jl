@inline maybe_particles(x) = x
@inline maybe_particles(p::AbstractParticles) = p.particles

"""
    register_primitive(f, eval=eval)

Register both single and multi-argument function so that it works with particles. If you want to register functions from within a module, you must pass the modules `eval` function.
"""
function register_primitive(ff, eval=eval)
    register_primitive_multi(ff, eval)
    register_primitive_single(ff, eval)
end

"""
    register_primitive_multi(ff, eval=eval)

Register a multi-argument function so that it works with particles. If you want to register functions from within a module, you must pass the modules `eval` function.
"""
function register_primitive_multi(ff, eval=eval)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    # for PT in (:Particles, :StaticParticles)
    #     eval(quote
    #         function ($m.$f)(p::$PT{T,N},a::Real...) where {T,N}
    #             res = ($m.$f).(p.particles, MonteCarloMeasurements.maybe_particles.(a)...) # maybe_particles introduced to handle >2 arg operators
    #             return $PT{eltype(res),N}(res)
    #         end
    #         function ($m.$f)(a::Real,p::$PT{T,N}) where {T,N}
    #             res = map(x->($m.$f)(a,x), p.particles)
    #             return $PT{eltype(res),N}(res)
    #         end
    #         function ($m.$f)(p1::$PT{T,N},p2::$PT{T,N}) where {T,N}
    #             res = map(($m.$f), p1.particles, p2.particles)
    #             return $PT{eltype(res),N}(res)
    #         end
    #         function ($m.$f)(p1::$PT{T,N},p2::$PT{S,N}) where {T,S,N} # Needed for particles of different float types :/
    #             res = map(($m.$f), p1.particles, p2.particles)
    #             return $PT{eltype(res),N}(res)
    #         end
    #     end)
    # end
    for PT in (:Particles, :StaticParticles)
        eval(quote
            function ($m.$f)(p::$PT{T,N},a::Real...) where {T,N}
                res = ($m.$f).(p.particles, MonteCarloMeasurements.maybe_particles.(a)...) # maybe_particles introduced to handle >2 arg operators
                return $PT{eltype(res),N}(res)
            end
            function ($m.$f)(a::Real,p::$PT{T,N}) where {T,N}
                res = ($m.$f).(a, p.particles)
                return $PT{eltype(res),N}(res)
            end
            function ($m.$f)(p1::$PT{T,N},p2::$PT{T,N}) where {T,N}
                res = ($m.$f).(p1.particles, p2.particles)
                return $PT{eltype(res),N}(res)
            end
            function ($m.$f)(p1::$PT{T,N},p2::$PT{S,N}) where {T,S,N} # Needed for particles of different float types :/
                res = ($m.$f).(p1.particles, p2.particles)
                return $PT{eltype(res),N}(res)
            end
        end)
    end
    # The code below is resolving some method ambiguities
    eval(quote
        function ($m.$f)(p1::StaticParticles{T,N},p2::Particles{T,N}) where {T,N}
            res = map(($m.$f), p1.particles, p2.particles)
            return StaticParticles{eltype(res),N}(res)
        end
        function ($m.$f)(p1::StaticParticles{T,N},p2::Particles{S,N}) where {T,S,N} # Needed for particles of different float types :/
            res = map(($m.$f), p1.particles, p2.particles)
            return StaticParticles{eltype(res),N}(res)
        end

        function ($m.$f)(p1::Particles{T,N},p2::StaticParticles{T,N}) where {T,N}
            res = map(($m.$f), p1.particles, p2.particles)
            return StaticParticles{eltype(res),N}(res)
        end
        function ($m.$f)(p1::Particles{T,N},p2::StaticParticles{S,N}) where {T,S,N} # Needed for particles of different float types :/
            res = map(($m.$f), p1.particles, p2.particles)
            return StaticParticles{eltype(res),N}(res)
        end
    end)
end

"""
    register_primitive_single(ff, eval=eval)

Register a single-argument function so that it works with particles. If you want to register functions from within a module, you must pass the modules `eval` function.
"""
function register_primitive_single(ff, eval=eval)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        eval(quote
            function ($m.$f)(p::$PT{T,N}) where {T,N}
                res = ($m.$f).(p.particles)
                return $PT{eltype(res),N}(res)
            end
        end)
    end
end
