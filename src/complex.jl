
for PT in (:Particles, :StaticParticles)
    @eval begin
        Base.promote_rule(::Type{Complex{S}}, ::Type{$PT{T,N}}) where {S<:Real,T<:Real,N} = Complex{$PT{promote_type(S,T),N}}

    end

end




"""
    ℂ2ℂ_function(f::Function, z::Complex{<:AbstractParticles})
applies `f : ℂ → ℂ ` to `z::Complex{<:AbstractParticles}`.
"""
function ℂ2ℂ_function(f::F, z::Complex{T}) where {F,T<:AbstractParticles}
    rz,iz = z.re,z.im
    s = map(1:length(rz.particles)) do i
        @inbounds f(complex(rz[i], iz[i]))
    end
    complex(T(real.(s)), T(imag.(s)))
end

function ℂ2ℂ_function!(f::F, s, z::Complex{T}) where {F,T<:AbstractParticles}
    rz,iz = z.re,z.im
    map!(s, 1:length(rz.particles)) do i
        @inbounds f(complex(rz[i], iz[i]))
    end
    complex(T(real.(s)), T(imag.(s)))
end

for ff in (sqrt, exp, sin, cos)
    f = nameof(ff)
    @eval Base.$f(z::Complex{<: AbstractParticles}) = ℂ2ℂ_function($f, z)
    @eval $(Symbol(f,:!))(s, z::Complex{<: AbstractParticles}) = ℂ2ℂ_function!($f, s, z)
end

function Base.:(/)(a::Complex{T}, b::Complex{T}) where T<:AbstractParticles
    are = real(a); aim = imag(a); bre = real(b); bim = imag(b)
    if mean(abs(bre)) <= mean(abs(bim))
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
