abstract type AbstractParticles{T,N} <: Real end
struct Particles{T,N} <: AbstractParticles{T,N}
    particles::Vector{T}
end

struct StaticParticles{T,N} <: AbstractParticles{T,N}
    particles::SArray{Tuple{N}, T, 1, N}
end


Particles(N::Integer = 100) = Particles{Float64,N}(randn(N))
StaticParticles(N::Integer = 100) = StaticParticles{Float64,N}(SVector{N,Float64}(randn(N)))

excluded_functions = [:fill, :|>, :<:, :display, :show, :promote, :promote_rule, :promote_type, :size, :length, :ndims, :convert, :isapprox, :≈, :<, :(<=), :(==), :zeros, :zero, :eltype, :getproperty, :fieldtype, :rand, :randn]
functions_to_extend = setdiff(names(Base), excluded_functions)

for PT in (:Particles, :StaticParticles)
    @eval function Base.show(io::IO, p::$(PT){T,N}) where {T,N}
        sPT = string($PT)
        if ndims(T) < 1
            print(io, N, " $sPT: ", round(mean(p), digits=3), " ± ", round(std(p), digits=3))
        else
            print(io, N, " $sPT with mean ", round.(mean(p), digits=3), " and std ", round.(sqrt.(diag(cov(p))), digits=3))
        end
    end

    @eval $PT(v::Vector) = $PT{eltype(v),length(v)}(v)
    @eval $PT{T,N}(p::$PT{T,N}) where {T,N} = p

    @eval function $PT(d::Distribution, N=100)
        v = rand(d,N)
        $PT{eltype(v),N}(v)
    end

    for fs in functions_to_extend
        ff = @eval $fs
        ff isa Function || continue
        isempty(methods(ff)) && continue # Sort out intrinsics and builtins
        f = nameof(ff)
        if !isempty(methods(ff, (Real,Real)))
            @eval function Base.$f(p::$PT{T,N},a::Real...) where {T,N}
                $PT{T,N}(map(x->$f(x,a...), p.particles))
            end
            @eval function Base.$f(a::Real,p::$PT{T,N}) where {T,N}
                $PT{T,N}(map(x->$f(a,x), p.particles))
            end
            @eval function Base.$f(p1::$PT{T,N},p2::$PT{T,N}) where {T,N}
                $PT{T,N}(map($f, p1.particles, p2.particles))
            end
        end
    end

    @forward @eval($PT).particles Statistics.mean, Statistics.cov, Statistics.var, Statistics.std
    @forward @eval($PT).particles Base.iterate, Base.getindex
    @eval Base.length(p::$PT{T,N}) where {T,N} = N
    @eval Base.ndims(p::$PT{T,N}) where {T,N} = ndims(T)

    @eval Base.eltype(::Type{$PT{T,N}}) where {T,N} = T
    @eval Base.promote_rule(::Type{S}, ::Type{$PT{T,N}}) where {S,T,N} = $PT{promote_type(S,T),N}
    @eval Base.promote_rule(::Type{Complex}, ::Type{$PT{T,N}}) where {T,N} = Complex{$PT{T,N}}
    @eval Base.convert(::Type{$PT{T,N}}, f::Real) where {T,N} = $PT{T,N}(fill(T(f),N))
    @eval Base.convert(::Type{$PT{T,N}}, f::$PT{S,N}) where {T,N,S} = $PT{promote_type(T,S),N}($PT{promote_type(T,S),N}(f))
    # @eval Base.convert(::Type{S}, p::$PT{T,N}) where {S<:Real,T,N} = S(mean(p)) # Not good to define this
    @eval Base.zeros(::Type{$PT{T,N}}, dim::Integer) where {T,N} = [$PT(zeros(eltype(T),N)) for d = 1:dim]
    @eval Base.zero(::Type{$PT{T,N}}) where {T,N} = $PT(zeros(eltype(T),N))
    @eval Base.complex(::Type{$PT{T,N}}) where {T,N} = Complex{$PT{T,N}}
    @eval Base.isfinite(p::$PT{T,N}) where {T,N} = isfinite(mean(p))
    @eval Base.round(p::$PT{T,N}, r::RoundingMode; kwargs...) where {T,N} = round(mean(p), r; kwargs...)
    # @eval Base.AbstractFloat(p::$PT) = mean(p) # Not good to define this



    for ff in (*,+,-,/,sin,cos,tan,zero,sign,abs,sqrt)
        f = nameof(ff)
        @eval function (Base.$f)(p::$PT)
            $PT(map($f, p.particles))
        end
    end


end
Distributions.Normal(p::AbstractParticles) = Normal(mean(p), std(p))
Distributions.MvNormal(p::AbstractParticles) = MvNormal(mean(p), cov(p))

Base.:(==)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = p1.particles == p2.particles
Base.:(!=)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = p1.particles != p2.particles
Base.:<(a::Real,p::AbstractParticles) = a < mean(p)
Base.:<(p::AbstractParticles,a::Real) = mean(p) < a
Base.:<(p::AbstractParticles, a::AbstractParticles, lim=2) = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim
Base.:(<=)(p::AbstractParticles{T,N}, a::AbstractParticles{T,N}, lim::Real=2) where {T,N} = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim

Base.:≈(a::Real,p::AbstractParticles, lim=2) = abs(mean(p)-a)/std(p) < lim
Base.:≈(p::AbstractParticles, a::Real, lim=2) = abs(mean(p)-a)/std(p) < lim
Base.:≈(p::AbstractParticles, a::AbstractParticles, lim=2) = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim

Base.:!(p::AbstractParticles) = all(p.particles .== 0)


≲(a::Real,p::AbstractParticles,lim=2) = (mean(p)-a)/std(p) > lim
≲(p::AbstractParticles,a::Real,lim=2) = (a-mean(p))/std(p) > lim
≳(a::Real,p::AbstractParticles,lim=2) = ≲(a,p,lim)
≳(p::AbstractParticles,a::Real,lim=2) = ≲(p,a,lim)


@recipe function plot(p::AbstractParticles)
    seriestype --> :histogram
    @series p.particles
end
