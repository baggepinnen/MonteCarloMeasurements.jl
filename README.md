# MonteCarloMeasurements

[![Build Status](https://travis-ci.com/baggepinnen/MonteCarloMeasurements.jl.svg?branch=master)](https://travis-ci.com/baggepinnen/MonteCarloMeasurements.jl)

This package provides a type `Particles` that represents a distribution of a floating point number, kind of like the type `Measurement` from Measurements.jl. The difference is that `Particles` represent the distribution using a collection of unweighted particles, and can thus represent arbitrary distributions. The goal is to have a number of this type behave just as any other number while partaking in calculations. After a calculation, the `mean`, `std` etc. can be extracted from the number using the corresponding functions. `Particles` also interact with Distributions.jl, so that you can call, e.g., `Normal(p)` and get back a `Normal` type from distributions.

The most basic constructor of `Particles` acts more or less like `randn(N)`, i.e., it creates a particle cloud with distribution `Normal(0,1)`. To create a particle cloud with distribution `Normal(μ,σ)`, you can call `μ + σ*Particles(N)`, or `Particles(Normal(μ,σ),N)`. This last constructor works with any distribution from which one can sample.
