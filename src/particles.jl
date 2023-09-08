"""
    μ ± σ

Creates $DEFAULT_NUM_PARTICLES `Particles` with mean `μ` and std `σ`. It can also be used as a unary operator, a mean of 0 is then used with std `σ`.
If `μ` is a vector, the constructor `MvNormal` is used, and `σ` is thus treated as std if it's a scalar, and variances if it's a matrix or vector.
See also [`∓`](@ref), [`..`](@ref)
"""
±

"""
    μ ∓ σ

Creates $DEFAULT_STATIC_NUM_PARTICLES `StaticParticles` with mean `μ` and std `σ`. It can also be used as a unary operator, a mean of 0 is then used with std `σ`.
If `μ` is a vector, the constructor `MvNormal` is used, and `σ` is thus treated as std if it's a scalar, and variances if it's a matrix or vector.
See also [`±`](@ref), [`⊗`](@ref)
"""
∓


±(μ::Real,σ) = Particles{promote_type(float(typeof(μ)),float(typeof(σ))),DEFAULT_NUM_PARTICLES}(systematic_sample(DEFAULT_NUM_PARTICLES,Normal(μ,σ); permute=true))
±(μ::AbstractVector,σ) = Particles(DEFAULT_NUM_PARTICLES, MvNormal(μ, σ))
±(σ) = zero(σ) ± σ
∓(μ::Real,σ) = StaticParticles{promote_type(float(typeof(μ)),float(typeof(σ))),DEFAULT_STATIC_NUM_PARTICLES}(systematic_sample(DEFAULT_STATIC_NUM_PARTICLES,Normal(μ,σ); permute=true))
∓(μ::AbstractVector,σ) = StaticParticles(DEFAULT_STATIC_NUM_PARTICLES, MvNormal(μ, σ))
∓(σ) = zero(σ) ∓ σ

"""
    a .. b

Creates $DEFAULT_NUM_PARTICLES `Particles` with a `Uniform` distribution between `a` and `b`.
See also [`±`](@ref), [`⊗`](@ref)
"""
(..)(a,b) = Particles{float(promote_type(eltype(a), eltype(b))), DEFAULT_NUM_PARTICLES}(Random.GLOBAL_RNG, Uniform(a,b))


"""
    a ⊠ Distribution()

Multiplies `a` by $DEFAULT_NUM_PARTICLES `Particles` sampled from a specified `::Distribution`.
Shorthand for `a * Particles(Distribution())`, e.g., `a ⊠ Gamma(1)`.
"""
⊠(a,d::Distribution) = a * Particles{eltype(d), DEFAULT_NUM_PARTICLES}(Random.GLOBAL_RNG, d)

"""
    a ⊞ Distribution()

Adds $DEFAULT_NUM_PARTICLES `Particles` sampled from a specified `::Distribution` to `a`.
Shorthand for `a + Particles(Distribution())`, e.g., `1 ⊞ Binomial(3)`.
"""
⊞(a,d::Distribution) = a + Particles{eltype(d), DEFAULT_NUM_PARTICLES}(Random.GLOBAL_RNG, d)

"""
    ⊗(μ,σ) = outer_product(Normal.(μ,σ))

See also [`outer_product`](@ref), [`±`](@ref)
"""
⊗(μ,σ) = outer_product(Normal.(μ,σ))

"""
    p = outer_product([rng::AbstractRNG,] dists::Vector{<:Distribution}, N=100_000)

Creates a multivariate systematic sample where each dimension is sampled according to the corresponding univariate distribution in `dists`. Returns `p::Vector{Particles}` where each Particles has a length approximately equal to `N`.
The particles form the outer product between `d` systematically sampled vectors with length given by the d:th root of N, where `d` is the length of `dists`, All particles will be independent and have marginal distributions given by `dists`.

See also `MonteCarloMeasurements.⊗`
"""
function outer_product(rng::AbstractRNG, dists::AbstractVector{<:Distribution}, N=100_000)
    d = length(dists)
    N = floor(Int,N^(1/d))
    dims = map(dists) do dist
        v = systematic_sample(rng,N,dist; permute=true)
    end
    cart_prod = vec(collect(Iterators.product(dims...)))
    p = map(1:d) do i
        Particles(getindex.(cart_prod,i))
    end
end
function outer_product(dists::AbstractVector{<:Distribution}, N=100_000)
    return outer_product(Random.GLOBAL_RNG, dists, N)
end

# StaticParticles(N::Integer = DEFAULT_NUM_PARTICLES; permute=true) = StaticParticles{Float64,N}(SVector{N,Float64}(systematic_sample(N, permute=permute)))


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
"""
    shortform(p::AbstractParticles)
Return a short string describing the type
"""
shortform(p::Particles) = "Part"
shortform(p::StaticParticles) = "SPart"
function to_num_str(p::AbstractParticles{T}, d=3, ds=d-1) where T
    s = pstd(p)
    # TODO: be smart and select sig digits based on s
    if T <: AbstractFloat && s < eps(p)
        string(round(pmean(p), sigdigits=d))
    else
        string(round(pmean(p), sigdigits=d), " ± ", round(s, sigdigits=ds))
    end
end


function Base.show(io::IO, p::AbstractParticles{T,N}) where {T,N}
    print(io, to_num_str(p, 3))
end

function Base.show(io::IO, ::MIME"text/plain", p::AbstractParticles{T,N}) where {T,N}
    sPT = MonteCarloMeasurements.shortform(p)
    compact = get(io, :compact, false)
    if compact
        print(io, MonteCarloMeasurements.to_num_str(p, 6, 3))
    else
        print(io, MonteCarloMeasurements.to_num_str(p, 6, 3), " $(typeof(p))\n")
    end
end

Base.show(io::IO, z::Complex{PT}) where PT <: AbstractParticles =
    show(io, MIME"text/plain"(), z)
function Base.show(io::IO, ::MIME"text/plain", z::Complex{PT}) where PT <: AbstractParticles
    r, i = reim(z)
    compact = get(io, :compact, false)
    print(io, "(")
    show(io, r)
    print(io, ")")
    if pmaximum(i) < 0
        i = -i
        print(io, compact ? "-" : " - ")
    else
        print(io, compact ? "+" : " + ")
    end
    print(io, "(")
    show(io, i)
    print(io, ")")
    print(io, "im")
end

# function Base.show(io::IO, p::MvParticles)
#     sPT = shortform(p)
#     print(io, "(", N, " $sPT with mean ", round.(mean(p), sigdigits=3), " and std ", round.(sqrt.(diag(cov(p))), sigdigits=3),")")
# end
for mime in (MIME"text/x-tex", MIME"text/x-latex")
    @eval function Base.show(io::IO, ::$mime, p::AbstractParticles)
        print(io, "\$"); show(io, p); print("\$")
    end

    @eval function Base.show(io::IO, ::$mime, z::Complex{<:AbstractParticles})
        print(io, "\$")
        r, i = reim(z)
        compact = get(io, :compact, false)
        print(io, "(")
        show(io, r)
        print(io, ")")
        if pmaximum(i) < 0
            i = -i
            print(io, compact ? "-" : " - ")
        else
            print(io, compact ? "+" : " + ")
        end
        print(io, "(")
        show(io, i)
        print(io, ")")
        print(io, "i")
        print("\$")
    end
end


# Two-argument functions
# foreach(register_primitive_binop, [+,-,*,/,//,^])
foreach(register_primitive_multi, [+,-,*,/,//,^,max,min,mod,mod1,atan,atand,add_sum,hypot,clamp])
# One-argument functions
foreach(register_primitive_single, [+,-,
exp,exp2,exp10,expm1,
log,log10,log2,log1p,
sin,cos,tan,sind,cosd,tand,sinh,cosh,tanh,
asin,acos,atan,asind,acosd,atand,asinh,acosh,atanh,
zero,sign,abs,sqrt,rad2deg,deg2rad,float])

MvParticles(x::AbstractVector{<:AbstractArray{<:Number}}) = Particles(copy(reduce(hcat, x)'))
MvParticles(v::AbstractVector{<:Number}) = Particles(v)


function MvParticles(v::AbstractVector{<:Tuple})
    Particles.([getindex.(v,i) for i in 1:length(v[1])])
end

function MvParticles(s::Vector{NamedTuple{vs, T}}) where {vs, T}
    nt = NamedTuple()
    for k in keys(s[1])
        nt = merge(nt, [k => MvParticles(getproperty.(s,k))])
    end
    nt
end

function _finish_individuals(::Type{PT}, N, individuals::AbstractArray{<:Tuple}, p) where PT
    ntuple(length(individuals[1])) do ti
        RT = eltype(first(individuals)[ti])
        PRT = PT{RT,N}
        no = length(individuals[1][ti])
        out = Vector{PRT}(undef, no)
        for i = 1:no
            out[i] = PRT(getindex.(getindex.(individuals,ti), i))
        end
        reshape(out, size(individuals[1][ti]))
    end
end

function _finish_individuals(::Type{PT}, ::Val{N}, individuals, p) where {PT, N}
    RT = eltype(eltype(individuals))
    PRT = PT{RT,N}
    out = similar(p, PRT)
    for i = 1:length(p)
        out[i] = PRT(getindex.(individuals,i))
    end
    reshape(out, size(p))
end

for PT in ParticleSymbols
    # Constructors
    @eval begin

        """
            ℝⁿ2ℝⁿ_function(f::Function, p::AbstractArray{T})

        Helper function for performing uncertainty propagation through vector-valued functions with vector inputs.
        Applies  `f : ℝⁿ → ℝⁿ` to an array of particles. E.g., `Base.log(p::Matrix{<:AbstractParticles}) = ℝⁿ2ℝⁿ_function(log,p)`
        """
        function ℝⁿ2ℝⁿ_function(f::F, p::AbstractArray{$PT{T,N}}) where {F,T,N}
            individuals = map(1:nparticles(p[1])) do i
                f(vecindex.(p,i))
            end
            _finish_individuals($PT, Val{N}(), individuals, p)
        end

        function ℝⁿ2ℝⁿ_function(f::F, p::AbstractArray{$PT{T,N}}, p2::AbstractArray{$PT{T,N}}) where {F,T,N}
            individuals = map(1:nparticles(p[1])) do i
                f(vecindex.(p,i), vecindex.(p2,i))
            end
            _finish_individuals($PT, Val{N}(), individuals, p)
        end

        """
            ℝⁿ2ℂⁿ_function(f::Function, p::AbstractArray{T})

        Helper function for performing uncertainty propagation through complex-valued functions with vector inputs.
        Applies  `f : ℝⁿ → Cⁿ` to an array of particles. E.g., `LinearAlgebra.eigvals(p::Matrix{<:AbstractParticles}) = ℝⁿ2ℂⁿ_function(eigvals,p)`
        """
        function ℝⁿ2ℂⁿ_function(f::F, p::AbstractArray{$PT{T,N}}; kwargs...) where {F,T,N}
            individuals = map(1:nparticles(p[1])) do i
                f(vecindex.(p,i); kwargs...)
            end
            PRT = $PT{T,N}
            RT = eltype(eltype(individuals))
            if RT <: Complex
                CRT = Complex{PRT}
            else
                CRT = PRT
            end
            out = Array{CRT}(undef, size(individuals[1]))
            for i = eachindex(out)
                ind = getindex.(individuals,i)
                if RT <: Complex
                    out[i] = complex(PRT(real.(ind)), PRT(imag.(ind)))
                else
                    out[i] = PRT(ind)
                end
            end
            out
        end
        #
        # function ℝⁿ2ℂⁿ_function(f::F, p::AbstractArray{$PT{T,N}}, p2::AbstractArray{$PT{T,N}}) where {F,T,N}
        #     individuals = map(1:nparticles(p[1])) do i
        #         f(getindex.(p,i), getindex.(p2,i))
        #     end
        #     RT = eltype(eltype(individuals))
        #     @assert RT <: Complex
        #     PRT = $PT{T,N}
        #     CRT = Complex{PRT}
        #     out = similar(p, CRT)
        #     for i = 1:length(p)
        #         ind = getindex.(individuals,i)
        #         out[i] = complex(PRT(real.(ind)), PRT(imag.(ind)))
        #     end
        #     reshape(out, size(p))
        # end
    end
end

for ff in (var, std)
    f = nameof(ff)
    @eval function (Statistics.$f)(p::ParticleDistribution{<:AbstractParticles{T,N}}, args...; kwargs...) where {N,T}
        N == 1 && (return zero(T))
        $f(p.p.particles, args...;kwargs...)
    end
    pname = Symbol("p"*string(f))
    @eval function ($pname)(p::AbstractParticles{T,N}, args...; kwargs...) where {N,T}
        N == 1 && (return zero(T))
        $f(p.particles, args...;kwargs...)
    end
end
# Instead of @forward
# TODO: convert all these to operate on ParticleDistribution
for ff in [Statistics.mean, Statistics.cov, Statistics.median, Statistics.quantile, Statistics.middle, Base.iterate, Base.extrema, Base.minimum, Base.maximum]
    f = nameof(ff)
    m = Base.parentmodule(ff)
    @eval ($m.$f)(p::ParticleDistribution, args...; kwargs...) = ($m.$f)(p.p.particles, args...; kwargs...)
    pname = Symbol("p"*string(f))
    @eval ($pname)(p::AbstractParticles, args...; kwargs...) = ($m.$f)(p.particles, args...; kwargs...)
    @eval ($pname)(p::Number, args...; kwargs...) = ($m.$f)(p, args...; kwargs...)
end

for PT in ParticleSymbols

    @eval begin
        # Base.length(::Type{$PT{T,N}}) where {T,N} = N
        # Base.eltype(::Type{$PT{T,N}}) where {T,N} = $PT{T,N} # TODO: remove

        Base.convert(::Type{StaticParticles{T,N}}, p::$PT{T,N}) where {T,N} = StaticParticles(p.particles)
        Base.convert(::Type{$PT{T,N}}, f::Real) where {T,N} = $PT{T,N}(fill(T(f),N))
        Base.convert(::Type{$PT{T,N}}, f::$PT{S,N}) where {T,N,S} = $PT{promote_type(T,S),N}(convert.(promote_type(T,S),f.particles))
        function Base.convert(::Type{S}, p::$PT{T,N}) where {S<:ConcreteFloat,T,N}
            N == 1 && (return S(p.particles[1]))
            pstd(p) < eps(S) || throw(ArgumentError("Cannot convert a particle distribution to a float if not all particles are the same."))
            return S(p.particles[1])
        end
        function Base.convert(::Type{S}, p::$PT{T,N}) where {S<:ConcreteInt,T,N}
            isinteger(p) || throw(ArgumentError("Cannot convert a particle distribution to an int if not all particles are the same."))
            return S(p.particles[1])
        end
        Base.zeros(::Type{$PT{T,N}}, dim::Integer) where {T,N} = [$PT{T,N}(zeros(eltype(T),N)) for d = 1:dim]
        Base.zero(::Type{$PT{T,N}}) where {T,N} = $PT{T,N}(zeros(eltype(T),N))
        Base.isfinite(p::$PT{T,N}) where {T,N} = isfinite(pmean(p))
        Base.round(p::$PT{T,N}, r::RoundingMode, args...; kwargs...) where {T,N} = $PT{T,N}(round.(p.particles, r, args...; kwargs...))
        Base.round(::Type{S}, p::$PT{T,N}, args...; kwargs...) where {S,T,N} = $PT{S,N}(round.(S, p.particles, args...; kwargs...))
        function Base.AbstractFloat(p::$PT{T,N}) where {T,N}
            N == 1 && (return p.particles[1])
            pstd(p) < eps(T) || throw(ArgumentError("Cannot convert a particle distribution to a number if not all particles are the same."))
            return p.particles[1]
        end
        Base.rem(p1::$PT{T,N}, p2::$PT{T,N}, args...) where {T,N} = $PT{T,N}(Base.rem.(p1.particles, p2.particles, args...))
        Base.div(p1::$PT{T,N}, p2::$PT{T,N}, args...) where {T,N} = $PT{T,N}(Base.div.(p1.particles, p2.particles, args...))

        """
            union(p1::AbstractParticles, p2::AbstractParticles)

        A `Particles` containing all particles from both `p1` and `p2`. Note, this will be twice as long as `p1` or `p2` and thus of a different type.
        `pu = Particles([p1.particles; p2.particles])`
        """
        function Base.union(p1::$PT{T,NT},p2::$PT{T,NS}) where {T,NT,NS}
            $PT{T,NT+NS}([p1.particles; p2.particles])
        end

        """
            intersect(p1::AbstractParticles, p2::AbstractParticles)

        A `Particles` containing all particles from the common support of `p1` and `p2`. Note, this will be of undetermined length and thus undetermined type.
        """
        function Base.intersect(p1::$PT,p2::$PT)
            mi = max(pminimum(p1),pminimum(p2))
            ma = min(pmaximum(p1),pmaximum(p2))
            f = x-> mi <= x <= ma
            $PT([filter(f, p1.particles); filter(f, p2.particles)])
        end

        function Base.:^(p::$PT{T,N}, i::Integer) where {T,N} # Resolves ambiguity
            res = p.particles.^i
            $PT{eltype(res),N}(res)
        end
        Base.:\(p::Vector{<:$PT}, p2::Vector{<:$PT}) = Matrix(p)\Matrix(p2) # Must be here to be most specific

        function LinearAlgebra.eigvals(p::Matrix{$PT{T,N}}; kwargs...) where {T,N} # Special case to propte types differently
            individuals = map(1:N) do i
                eigvals(vecindex.(p,i); kwargs...)
            end
            PRT = Complex{$PT{T,N}}
            out = Vector{PRT}(undef, length(individuals[1]))
            for i = eachindex(out)
                c = getindex.(individuals,i)
                out[i] = complex($PT{T,N}(real(c)),$PT{T,N}(imag(c)))
            end
            out
        end

    end

    # for XT in (:T, :($PT{T,N})), YT in (:T, :($PT{T,N})), ZT in (:T, :($PT{T,N}))
    #     XT == YT == ZT == :T && continue
    #     @eval function Base.muladd(x::$XT,y::$YT,z::$ZT) where {T<:Number,N}
    #          res = muladd.(maybe_particles(x),maybe_particles(y),maybe_particles(z))
    #          $PT{eltype(res),N}(res)
    #     end
    # end

    @eval function Base.muladd(x::$PT{T,N},y::$PT{T,N},z::$PT{T,N}) where {T<:Number,N}
        res = muladd.(x.particles,y.particles,z.particles)
        $PT{T,N}(res)
    end
    @eval function Base.muladd(x::T,y::$PT{T,N},z::$PT{T,N}) where {T<:Number,N}
        res = muladd.(x,y.particles,z.particles)
        $PT{T,N}(res)
    end
    @eval function Base.muladd(x::T,y::T,z::$PT{T,N}) where {T<:Number,N}
        res = muladd.(x,y,z.particles)
        $PT{T,N}(res)
    end

    @eval Base.promote_rule(::Type{S}, ::Type{$PT{T,N}}) where {S<:Number,T,N} = $PT{promote_type(S,T),N} # This is hard to hit due to method for real 3 lines down
    @eval Base.promote_rule(::Type{Bool}, ::Type{$PT{T,N}}) where {T,N} = $PT{promote_type(Bool,T),N}

    for PT2 in ParticleSymbols
        if PT == PT2
            @eval Base.promote_rule(::Type{$PT{S,N}}, ::Type{$PT{T,N}}) where {S,T,N} = $PT{promote_type(S,T),N}
        elseif any(==(:StaticParticles), (PT, PT2))
            @eval Base.promote_rule(::Type{$PT{S,N}}, ::Type{$PT2{T,N}}) where {S,T,N} = StaticParticles{promote_type(S,T),N}
        else
            @eval Base.promote_rule(::Type{$PT{S,N}}, ::Type{$PT2{T,N}}) where {S,T,N} = Particles{promote_type(S,T),N}
        end
    end

    @eval Base.promote_rule(::Type{<:AbstractParticles}, ::Type{$PT{T,N}}) where {T,N} = Union{}
end

# Base.length(p::AbstractParticles{T,N}) where {T,N} = N
Base.ndims(p::AbstractParticles{T,N}) where {T,N} = ndims(T)
Base.:\(H::MvParticles,p::AbstractParticles) = Matrix(H)\p.particles
# Base.:\(p::AbstractParticles, H) = p.particles\H
# Base.:\(p::MvParticles, H) = Matrix(p)\H
# Base.:\(H,p::MvParticles) = H\Matrix(p)

Base.Broadcast.broadcastable(p::AbstractParticles) = Ref(p)
# Base.setindex!(p::AbstractParticles, val, i::Integer) = setindex!(p.particles, val, i)
# Base.getindex(p::AbstractParticles, i::Integer) = getindex(p.particles, i)
# Base.getindex(v::MvParticles, i::Int, j::Int) = v[j][i] # Defining this methods screws with show(::MvParticles)

Base.Array(p::AbstractParticles) = p.particles
Base.Vector(p::AbstractParticles) = Array(p)

function Base.Array(v::Array{<:AbstractParticles})
    m = reduce(hcat, Array.(v))
    return reshape(m, size(m, 1), size(v)...)
end
Base.Matrix(v::MvParticles) = Array(v)

# function Statistics.var(v::MvParticles,args...;kwargs...) # Not sure if it's a good idea to define this. Is needed for when var(v::AbstractArray) is used
#     s2 = map(1:length(v[1])) do i
#         var(getindex.(v,i))
#     end
#     eltype(v)(s2)
# end

pmean(v::MvParticles) = pmean.(v)
pcov(v::MvParticles,args...;kwargs...) = cov(Matrix(v), args...; kwargs...)
pcor(v::MvParticles,args...;kwargs...) = cor(Matrix(v), args...; kwargs...)
pvar(v::MvParticles,args...; corrected = true, kwargs...) = sum(abs2, v)/(nparticles(v) - corrected)
Distributions.fit(d::Type{<:MultivariateDistribution}, p::MvParticles) = fit(d,Matrix(p)')
Distributions.fit(d::Type{<:Distribution}, p::AbstractParticles) = fit(d,p.particles)

Distributions.Normal(p::AbstractParticles) = Normal(pmean(p), pstd(p))
Distributions.MvNormal(p::MvParticles) = MvNormal(pmean(p), pcov(p))

meanstd(p::AbstractParticles) = pstd(p)/sqrt(nparticles(p))
meanvar(p::AbstractParticles) = pvar(p)/nparticles(p)

Base.:(==)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = p1.particles == p2.particles
Base.:(!=)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = p1.particles != p2.particles
function Base.hash(p::AbstractParticles, h::UInt)
    h = hash(p.particles, h)
    hash(typeof(p), h)
end


function zip_longest(a_,b_)
    a,b = maybe_particles(a_), maybe_particles(b_)
    l = max(length(a), length(b))
    Iterators.take(zip(Iterators.cycle(a), Iterators.cycle(b)), l)
end

function safe_comparison(a_, b_, op::F) where F
    a,b = maybe_particles(a_), maybe_particles(b_)
    all(((a,b),)->op(a,b), Iterators.product(extrema(a),extrema(b))) && (return true)
    !any(((a,b),)->op(a,b), Iterators.product(extrema(a),extrema(b))) && (return false)
    _comparison_error()
end

function do_comparison(a,b,op::F) where F
    mode = COMPARISON_MODE[]
    if mode === :reduction
        op(COMPARISON_FUNCTION[](a), COMPARISON_FUNCTION[](b))
    elseif mode === :montecarlo
        all(((a,b),)->op(a,b), zip_longest(a,b)) && return true
        !any(((a,b),)->op(a,b), zip_longest(a,b)) && return false
        _comparison_error()
    elseif mode === :safe
        safe_comparison(a,b,op)
    else
        error("Got unsupported comparison mode.")
    end
end

function _comparison_error()
    msg = "Comparison of uncertain values using comparison mode $(COMPARISON_MODE[]) failed. Comparison operators are not well defined for uncertain values. Call `unsafe_comparisons(true)` to enable comparison operators for particles using the current reduction function $(COMPARISON_FUNCTION[]). Change this function using `set_comparison_function(f)`. "
    if COMPARISON_MODE[] === :safe
        msg *= "For safety reasons, the default safe comparison function is maximally conservative and tests if the extreme values of the distributions fulfil the comparison operator."
    elseif COMPARISON_MODE[] === :montecarlo
        msg *= "For safety reasons, montecarlo comparison is conservative and tests if pairwise particles fulfil the comparison operator. If some do *and* some do not, this error is thrown. Consider if you can define a primitive function ([docs](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/overloading/#Overloading-a-new-function-1)) or switch to `unsafe_comparisons(:reduction)`"
    end

    error(msg)
end

function Base.:<(a::Real,p::AbstractParticles)
    do_comparison(a,p,<)
end
function Base.:<(p::AbstractParticles,a::Real)
    do_comparison(p,a,<)
end
function Base.:<(p::AbstractParticles, a::AbstractParticles)
    do_comparison(p,a,<)
end
function Base.:(<=)(p::AbstractParticles{T,N}, a::AbstractParticles{T,N}) where {T,N}
    do_comparison(p,a,<=)
end

"""
    p1 ≈ p2

Determine if two particles are not significantly different
"""
Base.:≈(p::AbstractParticles, a::AbstractParticles, lim=2) = abs(pmean(p)-pmean(a))/(2sqrt(pstd(p)^2 + pstd(a)^2)) < lim
function Base.:≈(a::Real,p::AbstractParticles, lim=2)
    m = pmean(p)
    s = pstd(p, mean=m)
    s == 0 && (return m == a)
    abs(pmean(p)-a)/pstd(p) < lim
end
function Base.:≈(p::AbstractParticles, a::Real, lim=2)
    m = pmean(p)
    s = pstd(p, mean=m)
    s == 0 && (return m == a)
    abs(pmean(p)-a)/pstd(p) < lim
end
Base.:≈(p::MvParticles, a::AbstractVector) = all(a ≈ b for (a,b) in zip(a,p))
Base.:≈(a::AbstractVector, p::MvParticles) = all(a ≈ b for (a,b) in zip(a,p))
Base.:≈(a::MvParticles, p::MvParticles) = all(a ≈ b for (a,b) in zip(a,p))
Base.:≉(a,b::AbstractParticles,lim=2) = !(≈(a,b,lim))
Base.:≉(a::AbstractParticles,b,lim=2) = !(≈(a,b,lim))
"""
    p1 ≉ p2

Determine if two particles are significantly different
"""
Base.:≉(a::AbstractParticles,b::AbstractParticles,lim=2) = !(≈(a,b,lim))

Base.sincos(x::AbstractParticles) = sin(x),cos(x)
Base.minmax(x::AbstractParticles,y::AbstractParticles) = (min(x,y), max(x,y))

Base.:!(p::AbstractParticles) = all(p.particles .== 0)

Base.isinteger(p::AbstractParticles) = all(isinteger, p.particles)
Base.iszero(p::AbstractParticles) = all(iszero, p.particles)
Base.iszero(p::AbstractParticles, tol) = abs(mean(p.particles)) < tol

≲(a,b,args...) = a < b
≲(a::Real,p::AbstractParticles,lim=2) = (pmean(p)-a)/pstd(p) > lim
≲(p::AbstractParticles,a::Real,lim=2) = (a-pmean(p))/pstd(p) > lim
≲(p::AbstractParticles,a::AbstractParticles,lim=2) = (pmean(a)-pmean(p))/(2sqrt(pstd(p)^2 + pstd(a)^2)) > lim
≳(a::Real,p::AbstractParticles,lim=2) = ≲(p,a,lim)
≳(p::AbstractParticles,a::Real,lim=2) = ≲(a,p,lim)
≳(p::AbstractParticles,a::AbstractParticles,lim=2) = ≲(a,p,lim)
Base.eps(p::Type{<:AbstractParticles{T,N}}) where {T,N} = eps(T)
Base.eps(p::AbstractParticles{T,N}) where {T,N} = eps(T)
Base.eps(p::AbstractParticles{<:Complex{T},N}) where {T,N} = eps(T)

Base.rtoldefault(::Type{<:AbstractParticles{T,N}}) where {T,N} = sqrt(eps(T))

LinearAlgebra.norm(x::AbstractParticles, args...) = abs(x)


Base.log(p::Matrix{<:AbstractParticles}) = ℝⁿ2ℂⁿ_function(log,p) # Matrix more specific than StridedMatrix used in Base.log
# LinearAlgebra.eigvals(p::Matrix{<:AbstractParticles}; kwargs...) = ℝⁿ2ℂⁿ_function(eigvals,p; kwargs...) # Replaced with implementation below
Base.exp(p::AbstractMatrix{<:AbstractParticles}) = ℝⁿ2ℝⁿ_function(exp, p)
LinearAlgebra.exp!(p::AbstractMatrix{<:AbstractParticles}) = ℝⁿ2ℝⁿ_function(LinearAlgebra.exp!, p)
LinearAlgebra.lyap(p1::Matrix{<:AbstractParticles}, p2::Matrix{<:AbstractParticles}) = ℝⁿ2ℝⁿ_function(lyap, p1, p2)
LinearAlgebra.hessenberg!(A::StridedMatrix{<: AbstractParticles}) = GenericSchur._hessenberg!(A)
Base.floatmin(::Type{<:AbstractParticles{T}}) where T = floatmin(T)
Base.floatmax(::Type{<:AbstractParticles{T}}) where T = floatmax(T)

## Particle BLAS

# pgemv is up to twice as fast as the naive way already for A(2,2)-A(20,20)
"""
    _pgemv(A, p::Vector{StaticParticles{T, N}}) where {T, N}

Perform `A*p::Vector{StaticParticles{T,N}` using BLAS matrix-matrix multiply. This function is automatically used when applicable and there is no need to call it manually.
"""
function _pgemv(
    A,
    p::Vector{StaticParticles{T,N}},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N}
    pm = reinterpret(T, p)
    M = reshape(pm, N, :)'
    AM = A * M
    reinterpret(StaticParticles{T,N}, vec(AM'))
end

Base.:*(A::Matrix{T}, p::Vector{StaticParticles{T,N}}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N} = _pgemv(A,p)


"""
    _pdot(v::Vector{T}, p::Vector{StaticParticles{T, N}}) where {T, N}

Perform `v'p::Vector{StaticParticles{T,N}` using BLAS matrix-vector multiply. This function is automatically used when applicable and there is no need to call it manually.
"""
function _pdot(
    v::AbstractVector{T},
    p::Vector{StaticParticles{T,N}},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N}
    pm = reinterpret(T, p)
    M = reshape(pm, N, :)
    Mv = M*v
    StaticParticles{T,N}(Mv)
end

LinearAlgebra.dot(v::AbstractVector{T}, p::Vector{StaticParticles{T,N}}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N} = _pdot(v,p)
LinearAlgebra.dot(p::Vector{StaticParticles{T,N}}, v::AbstractVector{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N} = _pdot(v,p)


function _paxpy!(
    a::T,
    x::Vector{StaticParticles{T,N}},
    y::Vector{StaticParticles{T,N}},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N}
    X = reinterpret(T, x)
    Y = reinterpret(T, y)
    LinearAlgebra.axpy!(a,X,Y)
    reinterpret(StaticParticles{T,N}, Y)
end

LinearAlgebra.axpy!(
    a::T,
    x::Vector{StaticParticles{T,N}},
    y::Vector{StaticParticles{T,N}},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N} = _paxpy!(a,x,y)





function LinearAlgebra.mul!(
    y::Vector{StaticParticles{T,N}},
    A::AbstractMatrix{T},
    b::Vector{StaticParticles{T,N}},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N}
    Bv = reinterpret(T, b)
    B = reshape(Bv, N, :)'
    # Y0 = A*B
    # reinterpret(StaticParticles{T,N}, vec(Y0'))
    Yv = reinterpret(T, y)
    Y = reshape(Yv, :, N)
    mul!(Y,A,B)
    reinterpret(StaticParticles{T,N}, vec(Y'))
end

function LinearAlgebra.mul!(
    y::Vector{Particles{T,N}},
    A::AbstractMatrix{T},
    b::Vector{Particles{T,N}},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64},N}

    B = Matrix(b)
    # Y = A*B
    Y = B*A' # This order makes slicing below more efficient
    @inbounds if isdefined(y, 1)
        for i in eachindex(y)
            @views y[i].particles .= Y[:,i]
        end
    else
        for i in eachindex(y)
            y[i] = Particles(Y[:,i])
        end
    end
    y
end


"""
    particle_dict2dict_vec(dict)

Take a dict that vaps keys to uncertain values, and return a vector of dicts where each dict has a single sample (particle) of the uncertain values. The length of the returned vector is the number of samples (particles) for all uncertain parameters. 
"""
function particle_dict2dict_vec(dict)
    # check the validity of uncertain parameters
    found_particle_numbers = Set{Int}()
    uncertain_parameters = Set{Base.keytype(dict)}()
    for (k, v) in dict
        if v isa AbstractParticles
            push!(found_particle_numbers, nparticles(v))
            push!(uncertain_parameters, k)
        end
    end
    if length(found_particle_numbers) > 1
        error("The number of samples (particles) for all uncertain parameters must be the same, but I found $(found_particle_numbers)")
    elseif isempty(found_particle_numbers)
        return [dict] # not much to do here
    end
    N = only(found_particle_numbers)
    map(1:N) do i
        Dict(k => vecindex(v, i) for (k, v) in dict)
    end
end