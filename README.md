# MonteCarloMeasurements

[![Build Status](https://travis-ci.com/baggepinnen/MonteCarloMeasurements.jl.svg?branch=master)](https://travis-ci.com/baggepinnen/MonteCarloMeasurements.jl)

This package provides two types `Particles <: Real` and `StaticParticles <: Real` that represents a distribution of a floating point number, kind of like the type `Measurement` from Measurements.jl. The difference compared to a `Measurement` is that `Particles` represent the distribution using a vector of unweighted particles, and can thus represent arbitrary distributions. The goal is to have a number of this type behave just as any other number while partaking in calculations. After a calculation, the `mean`, `std` etc. can be extracted from the number using the corresponding functions. `Particles` also interact with Distributions.jl, so that you can call, e.g., `Normal(p)` and get back a `Normal` type from distributions or `fit(Gamma, p)` to get a `Gamma`distribution.

The benefit of using this number type instead of manually calling a function `f` with perturbed inputs is that, at least in theory, each intermediate operation on a `Particles` can exploit SIMD, since it's performed over a vector. If the function `f` is called several times, however, the compiler might not be smart enough to SIMD the entire thing. An example
```julia
using BenchmarkTools
A = [Particles(1000) for i = 1:3, j = 1:3]
B = similar(A, Float64)
@btime qr($A)
  119.243 μs (257 allocations: 456.58 KiB)
@btime foreach(_->qr($B), 1:1000)
  3.916 ms (4000 allocations: 500.00 KiB)
```
that's about a 30-fold reduction in time, and the repeated `qr` didn't even store or handle the statistics of the result.
The type `StaticParticles` contains a statically sized, stack-allocated vector from StaticArrays.jl. This type is suitable if the number of particles is small, say < 500 ish.
```julia
A = [StaticParticles(100) for i = 1:3, j = 1:3]
B = similar(A, Float64)
@btime qr($(copy(A)))
  8.392 μs (16 allocations: 18.94 KiB)
@btime map(_->qr($B), 1:100);
  690.590 μs (403 allocations: 50.92 KiB)
# Wow that's over 80 times faster
# Bigger matrix
A = [StaticParticles(100) for i = 1:30, j = 1:30]
B = similar(A, Float64)
@btime qr($(copy(A)))
  1.823 ms (99 allocations: 802.63 KiB)
@btime map(_->qr($B), 1:100);
  75.068 ms (403 allocations: 2.11 MiB)
# 40 times faster
```
`StaticParticles` allocate much less memory than regular `Partricles`, but are more stressful for the compiler to handle.

## Constructors
The most basic constructor of `Particles` acts more or less like `randn(N)`, i.e., it creates a particle cloud with distribution `Normal(0,1)`. To create a particle cloud with distribution `Normal(μ,σ)`, you can call `μ + σ*Particles(N)`, or `Particles(Normal(μ,σ),N)`. This last constructor works with any distribution from which one can sample.
One can also call (`Particles/StaticParticles`)
- `Particles(v::Vector)` pre-sampled particles
- `Particles(d::Distribution, N::Int)` samples `N` particles from the distribution `d`.

**The default normal distribution is sampled systematically**, meaning that a single random number is drawn and used to seed the sample. This will reduce the variance of the sample. A side effect of this is that the particles are always sorted and a vector of `Particles` will exhibit strong correlations. If this is not desired, use the constructor `Particles(Normal(μ,σ),N)` instead.

If particles are plotted with `plot(p)`, a histogram is displayed. This requires Plots.jl.



## Multivariate particles
The constructors can be called with multivariate distributions, returning `v::Vector{Particle}` where particles are sampled from the desired multivariate distribution. Be wary of creating your own vector using the constructor `Particles(N)` due to the systematic sampling mentioned above, the particles would not be independent. Once `v` is propagated through a function `v2 = f(v)`, the results can be analyzed by asking for `mean(v2)` and `cov(v2)`, or by fitting a multivariate distribution, e.g., `MvNormal(v2)`.

A `v::Vector{Particle}` can be converted into a `Matrix` by calling `Matrix(v)` and this will have a size of `N × dim`.
