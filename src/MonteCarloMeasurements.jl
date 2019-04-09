module MonteCarloMeasurements

export Particles, ≲


using LinearAlgebra, Statistics, Random, Reexport, RecipesBase
using Lazy: @forward

@reexport using Distributions

struct Particles{T,N} <: Real
    particles::Vector{T}
end

function Base.show(io::IO, p::Particles{T,N}) where {T,N}
    if ndims(T) < 1
        print(io, N, " Particles: ", round(mean(p), digits=3), " ± ", round(std(p), digits=3))
    else
        print(io, N, " Particles with mean ", round.(mean(p), digits=3), " and std ", round.(sqrt.(diag(cov(p))), digits=3))
    end
end

function Particles(N::Integer = 100)
    Particles{Float64,N}(randn(N))
end

function Particles{T,N}(p::Particles{T,N}) where {T,N}
    p
end

function Particles(d::Distribution, N=100)
    v = rand(d,N)
    Particles{eltype(v),N}(v)
end


function Particles(v::Vector)
    Particles{eltype(v),length(v)}(v)
end

Distributions.Normal(p::Particles) = Normal(mean(p), std(p))
Distributions.MvNormal(p::Particles) = MvNormal(mean(p), cov(p))

excluded_functions = [:fill, :|>, :<:, :display, :show, :promote, :promote_rule, :promote_type, :size, :length, :ndims, :convert, :isapprox, :≈, :<, :(<=), :(==), :zeros, :zero, :eltype, :getproperty, :fieldtype, :rand, :randn]

for fs in setdiff(names(Base), excluded_functions)
    ff = @eval $fs
    ff isa Function || continue
    isempty(methods(ff)) && continue # Sort out intrinsics and builtins
    f = nameof(ff)
    if !isempty(methods(ff, (Real,Real)))
        @eval function Base.$f(p::Particles{T,N},a::Real...) where {T,N}
            Particles{T,N}(map(x->$f(x,a...), p.particles))
        end

        @eval function Base.$f(a::Real,p::Particles{T,N}) where {T,N}
            Particles{T,N}(map(x->$f(a,x), p.particles))
        end

        @eval function Base.$f(p1::Particles{T,N},p2::Particles{T,N}) where {T,N}
            Particles{T,N}(map($f, p1.particles, p2.particles))
        end
    end
end

@forward Particles.particles Statistics.mean, Statistics.cov, Statistics.std
@forward Particles.particles Base.iterate, Base.getindex
Base.length(p::Particles{T,N}) where {T,N} = N
Base.ndims(p::Particles{T,N}) where {T,N} = ndims(T)

Base.eltype(::Type{Particles{T,N}}) where {T,N} = T
Base.promote_rule(::Type{S}, ::Type{Particles{T,N}}) where {S,T,N} = Particles{promote_type(S,T),N}
Base.promote_rule(::Type{Complex}, ::Type{Particles{T,N}}) where {T,N} = Complex{Particles{T,N}}
Base.convert(::Type{Particles{T,N}}, f::Real) where {T,N} = Particles{T,N}(fill(T(f),N))
Base.convert(::Type{S}, p::Particles{T,N}) where {S<:Real,T,N} = S(mean(p))
Base.zeros(::Type{Particles{T,N}}, dim::Integer) where {T,N} = [Particles(zeros(eltype(T),N)) for d = 1:dim]
Base.zero(::Type{Particles{T,N}}) where {T,N} = Particles(zeros(eltype(T),N))
Base.complex(::Type{Particles{T,N}}) where {T,N} = Complex{Particles{T,N}}
Base.isfinite(p::Particles{T,N}) where {T,N} = isfinite(mean(p))
Base.round(p::Particles{T,N}, r::RoundingMode; kwargs...) where {T,N} = round(mean(p), r; kwargs...)
Base.AbstractFloat(p::Particles) = mean(p)


Base.:(==)(p1::Particles{T,N},p2::Particles{T,N}) where {T,N} = p1.particles == p2.particles
Base.:<(a::Real,p::Particles) = a < mean(p)
Base.:<(p::Particles,a::Real) = mean(p) < a
Base.:<(p::Particles, a::Particles, lim=2) = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim
Base.:(<=)(p::Particles{T,N}, a::Particles{T,N}, lim::Real=2) where {T,N} = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim

Base.:≈(a::Real,p::Particles, lim=2) = abs(mean(p)-a)/std(p) < lim
Base.:≈(p::Particles, a::Real, lim=2) = abs(mean(p)-a)/std(p) < lim
Base.:≈(p::Particles, a::Particles, lim=2) = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim

Base.:!(p::Particles) = all(p.particles .== 0)


≲(a::Real,p::Particles,lim=2) = (mean(p)-a)/std(p) > lim
≲(p::Particles,a::Real,lim=2) = (a-mean(p))/std(p) > lim
≳(a::Real,p::Particles,lim=2) = ≲(a,p,lim)
≳(p::Particles,a::Real,lim=2) = ≲(p,a,lim)

for ff in (*,+,-,/,sin,cos,tan,zero,sign,abs)
    f = nameof(ff)
    @eval function (Base.$f)(p::Particles)
        Particles(map($f, p.particles))
    end
end

@recipe function plot(p::Particles)
    seriestype --> :histogram
    @series p.particles
end

# InteractiveUtils, which, names, namesof
# TODO: Broadcast, Mutation, Code generation or define for all base functions, test on ControlSystems
# module Operators
# end
end # module
