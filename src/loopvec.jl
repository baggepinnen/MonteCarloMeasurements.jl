using SLEEFPirates, SIMDPirates, SpecialFunctions
for f in keys(LoopVectorization.SLEEFPiratesDict)
    # f = nameof(ff)
    pm = Base.parentmodule(eval(f))
    for PT in (:Particles, :StaticParticles)
        m,fe = LoopVectorization.SLEEFPiratesDict[f]
        eval(quote
            function ($pm.$f)(p::$PT{Float64,N}) where {Float64,N}
                res = map(($m.$fe), p.particles)
                return $PT{Float64,N}(res)
            end
            function ($pm.$f)(p::$PT{Float32,N}) where {Float32,N}
                res = map(($m.$fe), p.particles)
                return $PT{Float32,N}(res)
            end
        end)
    end
end



# for ff in (exp,)
#     f = nameof(ff)
#     m = Base.parentmodule(ff)
#
#     eval(quote
#     function ($m.$f)(p::StaticParticles{Float64,N}) where {Float64,N}
#         res = map((SLEEFPirates.$f), p.particles)
#         return StaticParticles{Float64,N}(res)
#     end
#     function ($m.$f)(p::StaticParticles{Float32,N}) where {Float32,N}
#         res = map((SLEEFPirates.$f), p.particles)
#         return StaticParticles{Float32,N}(res)
#     end
# end)
#
# end
#
#
# for ff in (log, sin, cos, asin, acos, atan) # tan is not faster
#     f = nameof(ff)
#     fs = Symbol(f,"_fast")
#     m = Base.parentmodule(ff)
#     eval(quote
#     function ($m.$f)(p::StaticParticles{Float64,N}) where {Float64,N}
#         res = map((SLEEFPirates.$fs), p.particles)
#         return StaticParticles{Float64,N}(res)
#     end
#     function ($m.$f)(p::StaticParticles{Float32,N}) where {Float32,N}
#         res = map((SLEEFPirates.$fs), p.particles)
#         return StaticParticles{Float32,N}(res)
#     end
# end)
#
# end
#
#
# for ff in [exp,exp2,exp10,expm1,
# log,log10,log2,log1p,
# sin,cos,tan,sind,cosd,tand,sinh,cosh,tanh,
# asin,acos,atan,asind,acosd,atand,asinh,acosh,atanh,sign,abs,sqrt]
#     f = nameof(ff)
#     m = Base.parentmodule(ff)
#         eval(quote
#             function ($m.$f)(p::Particles{Float64,N}) where {Float64,N}
#                 res = vmap(($m.$f), p.particles)
#                 return Particles{Float64,N}(res)
#             end
#             function ($m.$f)(p::Particles{Float32,N}) where {Float32,N}
#                 res = vmap(($m.$f), p.particles)
#                 return Particles{Float32,N}(res)
#             end
#         end)
#
# end
