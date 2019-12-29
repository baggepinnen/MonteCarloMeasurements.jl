
for ff in (exp,)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        eval(quote
            function ($m.$f)(p::$PT{Float64,N}) where {Float64,N}
                res = map((SLEEFPirates.$f), p.particles)
                return $PT{Float64,N}(res)
            end
        end)
    end
end


for ff in (exp,log)
    f = nameof(ff)
    m = Base.parentmodule(ff)
    for PT in (:Particles, :StaticParticles)
        eval(quote
            function ($m.$f)(p::$PT{Float32,N}) where {Float32,N}
                res = map((SLEEFPirates.$f), p.particles)
                return $PT{Float32,N}(res)
            end
        end)
    end
end
