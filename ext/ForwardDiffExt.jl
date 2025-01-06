module ForwardDiffExt

using MonteCarloMeasurements
using MonteCarloMeasurements: splitdef
import ForwardDiff
import ForwardDiff: Dual, value, partials, Partials

"""
    switch_representation(d::Dual{T, V, N}) where {T, V <: AbstractParticles, N}

Goes from Dual{Particles} to Particles{Dual}
"""
function switch_representation(d::Dual{T,V,N}) where {T,V<:AbstractParticles,N}
    part = partials(d)
    MonteCarloMeasurements.nakedtypeof(V)([Dual{T}(value(d).particles[i], ntuple(j->part[j].particles[i], N)) for i ∈ 1:nparticles(V)])
end

# function switch_representation(p::Particles{Dual{T,V,N},NP}) where {T,V<:AbstractParticles,N,NP}
#     Dual{T}(Particles(value.(p.particles)), Particles(partials.(p.particles)))
# end

const DualParticles = Dual{T,V,N} where {T,V<:AbstractParticles,N}
for ff in [maximum,minimum,std,var,cov,mean,median,quantile]
    f = nameof(ff)
    pname = Symbol("p"*string(f))
    m = Base.parentmodule(ff)
    @eval ($pname)(d::DualParticles) = ($pname)(switch_representation(d))
end

macro andreverse(ex)
    def = splitdef(ex)
    if haskey(def,:whereparams) && !isempty(def[:whereparams])
        quote
            $(esc(ex))
            $(esc(def[:name]))($(esc(def[:args][2])), $(esc(def[:args][1]))) where $(esc(def[:whereparams]...)) = $(esc(def[:body]))
        end
    else
        quote
            $(esc(ex))
            $(esc(def[:name]))($(esc(def[:args][2])), $(esc(def[:args][1]))) = $(esc(def[:body]))
        end
    end
end

for PT in (Particles, StaticParticles)
    @eval begin
        @andreverse function Base.:(*)(p::$PT, d::Dual{T}) where {T}
            Dual{T}($PT(p.particles .* value(d)), ntuple(i->$PT(p.particles .* partials(d)[i]) ,length(partials(d))))
        end

        @andreverse function Base.:(+)(p::$PT, d::Dual{T}) where {T}
            Dual{T}($PT(p.particles .+ value(d)), ntuple(i->$PT(0p.particles .+ partials(d)[i]) ,length(partials(d))))
        end

        function Base.:(-)(p::$PT, d::Dual{T}) where {T}
            Dual{T}($PT(p.particles .- value(d)), ntuple(i->$PT(0p.particles .- partials(d)[i]) ,length(partials(d))))
        end
        function Base.:(-)(d::Dual{T}, p::$PT) where {T}
            Dual{T}($PT(value(d) .- p.particles), ntuple(i->$PT(0p.particles .+ partials(d)[i]) ,length(partials(d))))
        end

        function Base.promote_rule(::Type{ForwardDiff.Dual{T,V,NP}}, ::Type{$PT{S, N}}) where {T, V, NP, S, N}
            VS = promote_type(V,S)
            Dual{T, $PT{VS, N}, NP}
            # Dual{T}($PT(fill(value(d), N)), ntuple(i->$PT(fill(partials(d)[i], N)) ,length(partials(d))))
        end

        # function Base.promote_rule(::Type{ForwardDiff.Dual{T, V, N}}, ::Type{$PT{T, N}}) where {T, V, N, T, N}
        #     Dual$PT{}
        # end
    end
end

# Base.hidigit(x::AbstractParticles, base) = Base.hidigit(mean(x), base) # To avoid stackoverflow in some printing situations
# Base.hidigit(x::Dual, base) = Base.hidigit(x.value, base) # To avoid stackoverflow in some printing situations
# Base.round(d::Dual, r::RoundingMode) = round(d.value,r)

end
