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

"""
    ℂ2ℂ_function(f::Function, z::Complex{<:AbstractParticles})

Helper function for uncertainty propagation through complex-valued functions of complex arguments.
applies `f : ℂ → ℂ ` to `z::Complex{<:AbstractParticles}`.
"""
function ℂ2ℂ_function(f::F, z::Complex{T}) where {F<:Union{Function,DataType},T<:AbstractParticles}
    rz,iz = z.re,z.im
    s = map(1:length(rz.particles)) do i
        @inbounds f(complex(rz.particles[i], iz.particles[i]))
    end
    complex(T(real.(s)), T(imag.(s)))
end

function ℂ2ℂ_function(f::F, z::Union{Complex{T},T}, a::R) where {F<:Union{Function,DataType},T<:AbstractParticles,R<:Real}
    rz,iz = z.re,z.im
    s = map(1:length(rz.particles)) do i
        @inbounds f(complex(rz.particles[i], iz.particles[i]),  a)
    end
    complex(T(real.(s)), T(imag.(s)))
end

function ℂ2ℂ_function(f::F, z::R, a::Union{Complex{S},S}) where {F<:Union{Function,DataType},S<:AbstractParticles,R<:Real}
    rz,iz = a.re,a.im
    s = map(1:length(rz.particles)) do i
        @inbounds f(z, complex(rz.particles[i], iz.particles[i]))
    end
    complex(S(real.(s)), S(imag.(s)))
end

function ℂ2ℂ_function(f::F, z::Union{Complex{T},T}, a::Union{Complex{S},S}) where {F<:Union{Function,DataType},T<:AbstractParticles,S<:AbstractParticles}
    rz,iz = z.re,z.im
    ra,ia = a.re,a.im

    s = map(1:length(rz.particles)) do i
        @inbounds f(complex(rz.particles[i], iz.particles[i]),  complex(ra.particles[i], ia.particles[i]))
    end
    complex(T(real.(s)), T(imag.(s)))
end

function ℂ2ℂ_function!(f::F, s, z::Complex{T}) where {F,T<:AbstractParticles}
    rz,iz = z.re,z.im
    map!(s, 1:length(rz.particles)) do i
        @inbounds f(complex(rz.particles[i], iz.particles[i]))
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

function Base.:(/)(a::Complex{T}, b::Complex{T}) where T<:AbstractParticles
    are = real(a); aim = imag(a); bre = real(b); bim = imag(b)
    if pmean(abs(bre)) <= pmean(abs(bim))
        if isinf(bre) && isinf(bim)
            r = sign(bre)/sign(bim)
        else
            r = bre / bim
        end
        den = bim + r*bre
        Complex((are*r + aim)/den, (aim*r - are)/den)
    else
        if isinf(bre) && isinf(bim)
            r = sign(bim)/sign(bre)
        else
            r = bim / bre
        end
        den = bre + r*bim
        Complex((are + aim*r)/den, (aim - are*r)/den)
    end
end

function Base.:(/)(are::T, b::Complex{T}) where T<:AbstractParticles
    aim = 0; bre = real(b); bim = imag(b)
    if pmean(abs(bre)) <= pmean(abs(bim))
        if isinf(bre) && isinf(bim)
            r = sign(bre)/sign(bim)
        else
            r = bre / bim
        end
        den = bim + r*bre
        Complex((are*r)/den, (-are)/den)
    else
        if isinf(bre) && isinf(bim)
            r = sign(bim)/sign(bre)
        else
            r = bim / bre
        end
        den = bre + r*bim
        Complex((are)/den, (-are*r)/den)
    end
end
