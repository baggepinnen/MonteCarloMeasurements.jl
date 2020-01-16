
for ff in (exp,)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        eval(quote
            @inline function ($m.$f)(p::$PT{Float64,N}) where {Float64,N}
                res = (SLEEFPirates.$f).(p.particles)
                return $PT{Float64,N}(res)
            end
            @inline function ($m.$f)(p::$PT{Float32,N}) where {Float32,N}
                res = (SLEEFPirates.$f).(p.particles)
                return $PT{Float32,N}(res)
            end
        end)
    end
end


for ff in (log, sin, cos, asin, acos, atan) # tan is not faster
    f = nameof(ff)
    fs = Symbol(f,"_fast")
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        eval(quote
        @inline function ($m.$f)(p::$PT{Float64,N}) where {Float64,N}
            res = (SLEEFPirates.$fs).(p.particles)
            return $PT{Float64,N}(res)
        end
        @inline function ($m.$f)(p::$PT{Float32,N}) where {Float32,N}
            res = (SLEEFPirates.$fs).(p.particles)
            return $PT{Float32,N}(res)
        end
        end)
    end
end
