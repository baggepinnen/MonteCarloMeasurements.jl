using MonteCarloMeasurements, Distributions
particles(m)
# (x = 0.00257 ± 1.0, y = Particles{Float64,1000}[1.0 ± 1.7, 1.0 ± 1.7, 1.0 ± 1.7, 0.998 ± 1.7, 1.0 ± 1.8, 1.0 ± 1.7, 0.998 ± 1.7, 0.997 ± 1.7, 0.999 ± 1.7, 0.996 ± 1.8])
# The way I get there... it's not so pretty. Let's start with μ and σ:

μ = Particles(1000,Normal(0,1))
# Part1000(0.000235 ± 1.0)

σ = Particles(1000,Normal(0,1))^2
# Part1000(1.008 ± 1.5)
# Surprisingly, this works just fine:

@unsafe d = Normal(μ,σ)
# Normal{Particles{Float64,1000}}(
# μ: 0.000235 ± 1.0
# σ: 1.01 ± 1.5
# )
# Now trying the obvious thing,

@unsafe x = rand(d)
# Part1000(-0.03248 ± 1.0)
# It thinks it works, but it's really horribly broken:

(x-μ)/σ
# Part1000(-0.03246 ± 2.6e-8)
# So I got it to work with a little helper function:

N = 1000
parts(x::Normal{P} where {P <: AbstractParticles}) = Particles(length(x.μ), Normal()) * x.σ + x.μ
# parts(x::Sampleable{F,S}) where {F,S} = Particles(N,x)
# parts(x::Integer) = parts(float(x))
# parts(x::Real) = parts(repeat([x],N))
# parts(x::AbstractArray) = Particles(x)
# parts(p::Particles) = p


@unsafe x = parts(d)
# Part1000(0.05171 ± 2.37)

(x-μ)/σ
# Part1000(0.0007729 ± 1.0)
# Much better.

# This is fine for one distribution, but most don't compose as nicely as a normal.

# Many distributions break entirely, seemingly because of argument checking in Distributions.jl:

@unsafe Bernoulli(Particles(1000,Beta(2,3)))
# ERROR: TypeError: non-boolean (Particles{Float64,1000}) used in boolean context

import MonteCarloMeasurements.particletype
struct ParticleDistribution{D}
    d::D
end

function ParticleDistribution(d::Type{<:Distribution}, p...)
    @unsafe ParticleDistribution(d(p...))
end
particletype(pd::ParticleDistribution) = particletype(getfield(pd.d,1))


function Base.rand(d::ParticleDistribution{D}) where D
    T,N = particletype(d)
    i = rand(1:N)
    d2 = MonteCarloMeasurements.replace_particles(d.d,P->P isa AbstractParticles, P->P[i])
    rand(d2)
end

pd = ParticleDistribution(Bernoulli, Particles(1000,Beta(2,3)))
@btime rand($pd)
@btime rand(Bernoulli(0.3))
@code_warntype rand(pd)
@code_warntype MonteCarloMeasurements.replace_particles(pd.d,P->P isa AbstractParticles, P->P[1])



using Distributions, MonteCarloMeasurements, LinearAlgebra, TransformVariables

z = Particles(1000,MvNormal(zeros(2,2) + I))
t = as((y=asℝ₊,p=as𝕀))


t([0.0,0.0])
# (y = 1.0, p = 0.5)

t([0.0,0.0]) |> inverse(t)


foreach(register_primitive, [<=, >=, <, >])

tz = t(z)
it = inverse(t)
@bymap it(tz)


using MonteCarloMeasurements, Test
p = 0 ± 1

@test MonteCarloMeasurements.Ngetter([p,p],p) == 500
@test MonteCarloMeasurements.Ngetter([p,p]) == 500

f(x) = 2x
f(x,y) = 2x + y

@test f(p) ≈ @bymap f(p)
@test f(p,p) ≈ @bymap f(p,p)

@bymap f(p,p)



##
using Zygote, MonteCarloMeasurements
p = 1 ± 0.1
f(x) = x^2
f'(p)
