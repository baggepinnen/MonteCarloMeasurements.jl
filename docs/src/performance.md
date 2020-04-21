# Performance tips
By using this package in favor of the naive way of performing Monte-Carlo propagation, you are already likely to to see a performance increase. Nevertheless, there are some things that can increase your performance further. Some of these tips are discussed in greater detail in the paper, ["MonteCarloMeasurements.jl: Nonlinear Propagation of Arbitrary Multivariate Distributions by means of Method Overloading"](https://arxiv.org/abs/2001.07625).

## Consider using `StaticParticles` and/or `sigmapoints`
If you want to propagate a small number of samples, less than about 300, [`StaticParticles`](@ref) are *much* faster than regular `Particles`. Above 300 samples, the compilation time starts exploding.

Using [`sigmapoints`](@ref) is a way to reduce the number of samples required, but they come with some caveats and also reduce the fidelity of the propagated distribution significantly.

## Use a smaller float type
While performing MonteCarlo propagation, it is very likely that the numerical precision offered by `Float64` is overkill, and that `Float32` will do just as well. All forms of `AbstractParticles` are generic with respect to the inner float type, and thus allow you to construct, e.g., `Particles{Float32,N}`. Since a large contributing factor to the speedup offered by this package comes from the utilization of SIMD instructions, switching to `Float32` will almost always give you a clean 2x performance improvement. Some examples on how to create `Float32` particles:
```@repl
using MonteCarloMeasurements # hide
1.0f0 ∓ 0.1f0
Particles(Normal(1f0, 0.1f0))
Particles(randn(Float32, 500))
```

## Faster `exp,log`
If the user manually loads the library [SLEEFPirates.jl](https://github.com/chriselrod/SLEEFPirates.jl), some functions are overloaded for Particles of `Float64,Float32` eltypes making these functions 2-16 times faster depending on the processor SIMD width.

## Try GPU particles
If you arepropagating a very large number of samples, say 10⁵-10⁷, you may want to try utilizing the GPU. Unfortunately, supporting GPU particles on the main branch has proven difficult since the CuArrays package is a heavy dependency and very hard to test on available CI infrastructure. We thus maintain a separate branch with this functionality. To install it, do
```julia
using Pkg
pkg"add MonteCarloMeasurements#gpu"
```
after which you will have access to a new particle type, `CuParticles`. These act just like ordinary particles, but perform all computations on the GPU.
