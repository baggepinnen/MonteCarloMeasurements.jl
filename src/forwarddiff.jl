import .ForwardDiff: Dual, value, partials, Partials # The dot in .ForwardDiff is an artefact of using Requires.jl

function switch_representation(d::Dual{T,V,N}) where {T,V<:AbstractParticles,N}
    part = partials(d)
    MonteCarloMeasurements.nakedtypeof(V)([Dual{T}(value(d)[i], ntuple(j->part[j][i], N)) for i ∈ 1:length(V)])
end

# function switch_representation(p::Particles{Dual{T,V,N},NP}) where {T,V<:AbstractParticles,N,NP}
#     Dual{T}(Particles(value.(p.particles)), Particles(partials.(p.particles)))
# end

const DualParticles = Dual{T,V,N} where {T,V<:AbstractParticles,N}
for ff in [maximum,minimum,std,var,cov,mean,median,quantile]
    f = nameof(ff)
    m = Base.parentmodule(ff)
    @eval ($m.$f)(d::DualParticles) = ($m.$f)(switch_representation(d))
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

# display(@macroexpand(@andreverse f(p::PT, d::Dual) = PT(p.particles .* Ref(d)))  |> prettify)

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
    end
end

# Base.hidigit(x::AbstractParticles, base) = Base.hidigit(mean(x), base) # To avoid stackoverflow in some printing situations
# Base.hidigit(x::Dual, base) = Base.hidigit(x.value, base) # To avoid stackoverflow in some printing situations
# Base.round(d::Dual, r::RoundingMode) = round(d.value,r)
