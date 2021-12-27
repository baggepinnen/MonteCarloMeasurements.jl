for PT in (:Particles, :StaticParticles)
    @eval begin
        Base.promote_rule(::Type{Complex{S}}, ::Type{$PT{T,N}}) where {S<:Real,T<:Real,N} = Complex{$PT{promote_type(S,T),N}}

    end

    for ff in (^,)
        f = nameof(ff)
        @eval Base.$f(z::Complex{$PT{T,N}}, x::Real) where {T,N} = ℂ2ℂ_function($f, z, x)
        @eval Base.$f(z::Real, x::Complex{$PT{T,N}}) where {T,N} = ℂ2ℂ_function($f, z, x)
        @eval Base.$f(z::Complex{$PT{T,N}}, x::Complex{$PT{T,N}}) where {T,N} = ℂ2ℂ_function($f, z, x)

        @eval Base.$f(z::Complex{$PT{T,N}}, x::Int) where {T,N} = ℂ2ℂ_function($f, z, x)
        @eval Base.$f(z::Int, x::Complex{$PT{T,N}}) where {T,N} = ℂ2ℂ_function($f, z, x)
    end

end


@inline maybe_complex_particles(x,i) = x
@inline maybe_complex_particles(p::AbstractParticles,i) = complex(real(p.particles[i]), imag(p.particles[i]))
@inline maybe_complex_particles(p::Complex{<:AbstractParticles},i) = complex(p.re.particles[i], p.im.particles[i])

"""
    ℂ2ℂ_function(f::Function, z::Complex{<:AbstractParticles})

Helper function for uncertainty propagation through complex-valued functions of complex arguments.
applies `f : ℂ → ℂ ` to `z::Complex{<:AbstractParticles}`.
"""
function ℂ2ℂ_function(f::F, z::Complex{T}) where {F<:Union{Function,DataType},T<:AbstractParticles}
    s = map(1:length(z.re.particles)) do i
        @inbounds f(maybe_complex_particles(z, i))
    end
    complex(T(real.(s)), T(imag.(s)))
end

function ℂ2ℂ_function(f::F, z::Union{Complex{T},T}, a::R) where {F<:Union{Function,DataType},T<:AbstractParticles,R<:Real}
    s = map(1:length(z.re.particles)) do i
        @inbounds f(maybe_complex_particles(z, i),  vecindex(a, i))
    end
    complex(T(real.(s)), T(imag.(s)))
end

function ℂ2ℂ_function(f::F, z::R, a::Complex{S}) where {F<:Union{Function,DataType},S<:AbstractParticles,R<:Real}
    s = map(1:length(a.re.particles)) do i
        @inbounds f(vecindex(z, i), maybe_complex_particles(a, i))
    end
    complex(S(real.(s)), S(imag.(s)))
end

function ℂ2ℂ_function(f::F, z::Complex{T}, a::Complex{S}) where {F<:Union{Function,DataType},T<:AbstractParticles,S<:AbstractParticles}
    s = map(1:length(z.re.particles)) do i
        @inbounds f(maybe_complex_particles(z, i),  maybe_complex_particles(a, i))
    end
    complex(T(real.(s)), T(imag.(s)))
end

# function ℂ2ℂ_function(f::F, z::Complex{T}, a::Complex{S}) where {F<:Union{Function,DataType},T<:AbstractParticles,S<:AbstractParticles}
#     out = deepcopy(z)
#     rp, ip = out.re.particles, out.im.particles
#     @inbounds for i = 1:length(z.re.particles)
#         res = f(maybe_complex_particles(z, i),  maybe_complex_particles(a, i))
#         rp[i] = res.re
#         ip[i] = res.im
#     end
#     out
# end

function ℂ2ℂ_function!(f::F, s, z::Complex{T}) where {F,T<:AbstractParticles}
    map!(s, 1:length(z.re.particles)) do i
        @inbounds f(maybe_complex_particles(z, i))
    end
    complex(T(real.(s)), T(imag.(s)))
end

for ff in (sqrt, exp, exp10, log, log10, sin, cos, tan)
    f = nameof(ff)
    @eval Base.$f(z::Complex{<: AbstractParticles}) = ℂ2ℂ_function($f, z)
    @eval $(Symbol(f,:!))(s, z::Complex{<: AbstractParticles}) = ℂ2ℂ_function!($f, s, z)
end

Base.isinf(p::Complex{<: AbstractParticles}) = isinf(real(p)) || isinf(imag(p))
Base.isfinite(p::Complex{<: AbstractParticles}) = isfinite(real(p)) && isfinite(imag(p))

function Base.:(/)(a::Union{T, Complex{T}}, b::Complex{T}) where T<:AbstractParticles
    ℂ2ℂ_function(/, a, b)
end

function Base.FastMath.div_fast(a::Union{T, Complex{T}}, b::Complex{T}) where T<:AbstractParticles
    ℂ2ℂ_function(Base.FastMath.div_fast, a, b)
end

for f in (:pmean, :pmaximum, :pminimum, :psum, :pstd, :pcov)
    @eval $f(p::Complex{<: AbstractParticles}) = Complex($f(p.re), $f(p.im))
end

function switch_representation(d::Complex{<:V}) where {V<:AbstractParticles}
    MonteCarloMeasurements.nakedtypeof(V)(complex.(d.re.particles, d.im.particles))
end

function complex_array(R::AbstractArray{Complex{V}}) where {V<:AbstractParticles}
    R = switch_representation.(R)
    permutedims(reinterpret(reshape, Float64, vec(R)), dims=())
end


function ℂⁿ2ℂⁿ_function(f::F, R::Matrix{<:Complex{<:AbstractParticles{T, N}}}) where {F, T, N}
    E = similar(R)
    for i in eachindex(E)
        E[i] = Complex(Particles(zeros(N)), Particles(zeros(N)))
    end
    r = zeros(Complex{T}, size(R)...)
    for n in 1:N
        for j in eachindex(R)
            r[j] = Complex(R[j].re.particles[n], R[j].im.particles[n]) 
        end
        e = f(r)
        for i in eachindex(e)
            E[i].re.particles[n] = e[i].re
            E[i].im.particles[n] = e[i].im
        end
    end
    E
end

Base.exp(R::Matrix{<:Complex{<:AbstractParticles}}) = ℂⁿ2ℂⁿ_function(exp, R)
Base.log(R::Matrix{<:Complex{<:AbstractParticles}}) = ℂⁿ2ℂⁿ_function(log, R)

function ℂⁿ2ℂ_function(f::F, D::Matrix{Complex{PT}}) where {F, PT <: AbstractParticles}
    D0 = similar(D, ComplexF64)
    parts = map(1:nparticles(D[1].re)) do i
        for j in eachindex(D0)
            D0[j] = Complex(D[j].re.particles[i], D[j].im.particles[i])
        end
        f(D0)
    end
    # PT = nakedtypeof(P)
    Complex(PT(getfield.(parts, :re)), PT(getfield.(parts, :im)))
end

LinearAlgebra.det(R::Matrix{<:Complex{<:AbstractParticles}}) = ℂⁿ2ℂ_function(det, R)

function LinearAlgebra.eigvals(R::Matrix{<:Complex{<:AbstractParticles{T, N}}}; kwargs...) where {T, N}
    E = Vector{Complex{Particles{T,N}}}(undef, size(R,1))
    for i in eachindex(E)
        E[i] = Complex(Particles(zeros(N)), Particles(zeros(N)))
    end
    r = zeros(Complex{T}, size(R)...)
    for n in 1:N
        for j in eachindex(R)
            r[j] = Complex(R[j].re.particles[n], R[j].im.particles[n]) 
        end
        e = eigvals!(r; kwargs...)
        for i in eachindex(e)
            E[i].re.particles[n] = e[i].re
            E[i].im.particles[n] = e[i].im
        end
    end
    E
end

function LinearAlgebra.svdvals(R::Matrix{<:Complex{<:AbstractParticles{T, N}}}; kwargs...) where {T, N}
    E = Vector{Particles{T,N}}(undef, size(R,1))
    for i in eachindex(E)
        E[i] = Particles(zeros(N))
    end
    r = zeros(Complex{T}, size(R)...)
    for n in 1:N
        for j in eachindex(R)
            r[j] = Complex(R[j].re.particles[n], R[j].im.particles[n]) 
        end
        e = svdvals!(r; kwargs...)
        for i in eachindex(e)
            E[i].particles[n] = e[i]
        end
    end
    E
end