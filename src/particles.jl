abstract type AbstractParticles{T,N} <: Real end
struct Particles{T,N} <: AbstractParticles{T,N}
    particles::Vector{T}
end

struct StaticParticles{T,N} <: AbstractParticles{T,N}
    particles::SArray{Tuple{N}, T, 1, N}
end

const MvParticles = Vector{<:AbstractParticles} # This can not be AbstractVector since it causes some methods below to be less specific than desired

±(μ::Real,σ) = μ + σ*Particles(500)
±(μ::AbstractVector,σ) = Particles(500, MvNormal(μ, σ))

# StaticParticles(N::Integer = 500; permute=true) = StaticParticles{Float64,N}(SVector{N,Float64}(systematic_sample(N, permute=permute)))

Base.Broadcast.broadcastable(p::Particles) = Ref(p)
Base.getindex(p::AbstractParticles, I::Integer...) = getindex(p.particles, I...)

function print_functions_to_extend()
    excluded_functions = [fill, |>, <, display, show, promote, promote_rule, promote_type, size, length, ndims, convert, isapprox, ≈, <, (<=), (==), zeros, zero, eltype, getproperty, fieldtype, rand, randn]
    functions_to_extend = setdiff(names(Base), Symbol.(excluded_functions))
    for fs in functions_to_extend
        ff = @eval $fs
        ff isa Function || continue
        isempty(methods(ff)) && continue # Sort out intrinsics and builtins
        f = nameof(ff)
        if !isempty(methods(ff, (Real,Real)))
            println(f, ",")
        end
    end
end

for PT in (:Particles, :StaticParticles)
    @forward @eval($PT).particles Statistics.mean, Statistics.cov, Statistics.var, Statistics.std, Statistics.median, Statistics.quantile, Statistics.middle
    @forward @eval($PT).particles Base.iterate, Base.extrema, Base.minimum, Base.maximum

    @eval begin
        function Base.show(io::IO, p::$(PT){T,N}) where {T,N}
            sPT = string($PT)
            if ndims(T) < 1
                print(io, N, " $sPT: ", round(mean(p), digits=3), " ± ", round(std(p), digits=3))
            else
                print(io, N, " $sPT with mean ", round.(mean(p), digits=3), " and std ", round.(sqrt.(diag(cov(p))), digits=3))
            end
        end

        $PT(v::Vector) = $PT{eltype(v),length(v)}(v)
        $PT{T,N}(p::$PT{T,N}) where {T,N} = p

        function $PT(N::Integer=500, d::Distribution=Normal(0,1); permute=true, systematic=true)
            if systematic
                v = systematic_sample(N,d; permute=permute)
            else
                v = rand(d, N)
            end
            $PT{eltype(v),N}(v)
        end

        function $PT(N::Integer, d::MultivariateDistribution)
            v = rand(d,N)' |> copy # For cache locality
            map($PT{eltype(v),N}, eachcol(v))
        end
    end
    # @eval begin

    for f in (:+,:-,:*,:/,://,:^, :max,:min,:minmax,:mod,:mod1,:atan)
        @eval begin
            function Base.$f(p::$PT{T,N},a::Real...) where {T,N}
                $PT{T,N}(map(x->$f(x,a...), p.particles))
            end
            function Base.$f(a::Real,p::$PT{T,N}) where {T,N}
                $PT{T,N}(map(x->$f(a,x), p.particles))
            end
            function Base.$f(p1::$PT{T,N},p2::$PT{T,N}) where {T,N}
                $PT{T,N}(map($f, p1.particles, p2.particles))
            end
            function Base.$f(p1::$PT{T,N},p2::$PT{S,N}) where {T,S,N} # Needed for particles of different float types :/
                $PT{promote_type(T,S),N}(map($f, p1.particles, p2.particles))
            end
        end
    end
    # end
    @eval begin
        Base.length(p::$PT{T,N}) where {T,N} = N
        Base.ndims(p::$PT{T,N}) where {T,N} = ndims(T)

        Base.eltype(::Type{$PT{T,N}}) where {T,N} = T
        Base.promote_rule(::Type{S}, ::Type{$PT{T,N}}) where {S,T,N} = $PT{promote_type(S,T),N}
        Base.promote_rule(::Type{Complex}, ::Type{$PT{T,N}}) where {T,N} = Complex{$PT{T,N}}
        Base.promote_rule(::Type{Complex{T}}, ::Type{$PT{T,N}}) where {T<:Real,N} = Complex{$PT{T,N}}
        Base.convert(::Type{$PT{T,N}}, f::Real) where {T,N} = $PT{T,N}(fill(T(f),N))
        Base.convert(::Type{$PT{T,N}}, f::$PT{S,N}) where {T,N,S} = $PT{promote_type(T,S),N}($PT{promote_type(T,S),N}(f))
        # Base.convert(::Type{S}, p::$PT{T,N}) where {S<:Real,T,N} = S(mean(p)) # Not good to define this
        Base.zeros(::Type{$PT{T,N}}, dim::Integer) where {T,N} = [$PT(zeros(eltype(T),N)) for d = 1:dim]
        Base.zero(::Type{$PT{T,N}}) where {T,N} = $PT(zeros(eltype(T),N))
        Base.isfinite(p::$PT{T,N}) where {T,N} = isfinite(mean(p))
        Base.round(p::$PT{T,N}, args...; kwargs...) where {T,N} = round(mean(p), args...; kwargs...)
        # Base.AbstractFloat(p::$PT) = mean(p) # Not good to define this


        Base.:^(p::$PT, i::Integer) = $PT(p.particles.^i) # Resolves ambiguity
        Base.:\(p::Vector{<:$PT}, p2::Vector{<:$PT}) = Matrix(p)\Matrix(p2) # Must be here to be most specific
    end

    for ff in (*,+,-,/,sin,cos,tan,zero,sign,abs,sqrt,asin,acos,atan,log,log10,log2,log1p,rad2deg)
        f = nameof(ff)
        @eval function (Base.$f)(p::$PT)
            $PT(map($f, p.particles))
        end
    end
    # Multivariate particles



end
Base.:\(H::MvParticles,p::AbstractParticles) = Matrix(H)\p.particles
# Base.:\(p::AbstractParticles, H) = p.particles\H
# Base.:\(p::MvParticles, H) = Matrix(p)\H
# Base.:\(H,p::MvParticles) = H\Matrix(p)


Base.Matrix(v::MvParticles) = reduce(hcat, getfield.(v,:particles))
Statistics.mean(v::MvParticles) = mean.(v)
Statistics.cov(v::MvParticles,args...;kwargs...) = cov(Matrix(v), args...; kwargs...)
Distributions.Normal(p::AbstractParticles) = Normal(mean(p), std(p))
Distributions.MvNormal(p::AbstractParticles) = MvNormal(mean(p), cov(p))
Distributions.MvNormal(p::MvParticles) = MvNormal(mean(p), cov(p))
Distributions.fit(d::Type{<:MultivariateDistribution}, p::MvParticles) = fit(d,Matrix(p)')
Distributions.fit(d::Type{<:Distribution}, p::AbstractParticles) = fit(d,p.particles)

Base.:(==)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = p1.particles == p2.particles
Base.:(!=)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = p1.particles != p2.particles
Base.:<(a::Real,p::AbstractParticles) = a < mean(p)
Base.:<(p::AbstractParticles,a::Real) = mean(p) < a
Base.:<(p::AbstractParticles, a::AbstractParticles, lim=2) = mean(p) < mean(a)
Base.:(<=)(p::AbstractParticles{T,N}, a::AbstractParticles{T,N}, lim::Real=2) where {T,N} = mean(p) <= mean(a)

Base.:≈(a::Real,p::AbstractParticles, lim=2) = abs(mean(p)-a)/std(p) < lim
Base.:≈(p::AbstractParticles, a::Real, lim=2) = abs(mean(p)-a)/std(p) < lim
Base.:≈(p::AbstractParticles, a::AbstractParticles, lim=2) = abs(mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim

Base.:!(p::AbstractParticles) = all(p.particles .== 0)


≲(a::Real,p::AbstractParticles,lim=2) = (mean(p)-a)/std(p) > lim
≲(p::AbstractParticles,a::Real,lim=2) = (a-mean(p))/std(p) > lim
≲(p::AbstractParticles,a::AbstractParticles,lim=2) = (mean(p)-mean(a))/(2sqrt(std(p)^2 + std(a)^2)) < lim
≳(a::Real,p::AbstractParticles,lim=2) = ≲(a,p,lim)
≳(p::AbstractParticles,a::Real,lim=2) = ≲(p,a,lim)
≳(p::AbstractParticles,a::AbstractParticles,lim=2) = ≲(p,a,lim)
Base.eps(p::Type{<:AbstractParticles{T,N}}) where {T,N} = eps(T)

function Base.sqrt(z::Complex{T}) where T <: AbstractParticles
    rz,iz = z.re,z.im
    s = map(1:length(rz.particles)) do i
        sqrt(complex(rz[i], iz[i]))
    end
    complex(T(real.(s)), T(imag.(s)))
end

@recipe function plot(p::AbstractParticles)
    seriestype --> :histogram
    @series p.particles
end

@recipe f(::Type{<:AbstractParticles}, p::AbstractParticles) = p.particles

function handle_args(p)
    length(p.args) < 2 && throw(ArgumentError("This function is called with at least two arguments (x, y, ..."))
    x,y = p.args[1:2]
    y isa MvParticles || throw(ArgumentError("The second argument must be a vector of some kind of Particles"))
    x,y
end
@userplot Errorbarplot
@recipe function plt(p::Errorbarplot)
    x,y = handle_args(p)
    q = length(p.args) >= 3 ? p.args[3] : 0.05
    m = mean.(y)
    lower = -(quantile.(y,q)-m)
    upper = quantile.(y,1-q)-m
    yerror := (lower,upper)
    x,m
end

@userplot MCplot
@recipe function plt(p::MCplot)
    x,y = handle_args(p)
    m = Matrix(y)'
    x,m
end

@userplot Ribbonplot
@recipe function plt(p::Ribbonplot)
    x,y = handle_args(p)
    q = length(p.args) >= 3 ? p.args[3] : 2.
    m = mean.(y)
    ribbon := q*std.(y)
    x,m
end

"""
    errorbarplot(x,y,[q=0.05])

Plots a vector of particles with error bars at quantile `q`
"""
errorbarplot

"""
    mcplot(x,y,[q=0.05])

Plots all trajectories represented by a vector of particles
"""
mcplot

"""
    ribbonplot(x,y,[q=2])

Plots a vector of particles with a ribbon representing `q*std(y)`. Default width is 2σ
"""
ribbonplot
