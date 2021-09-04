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
