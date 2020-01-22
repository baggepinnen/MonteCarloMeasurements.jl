# Supporting new functions


## Overloading a new function
If a method for `Particles` is not implemented for your function `yourfunc`, the pattern to register your function looks like this
```julia
register_primitive(yourfunc)
```
This defines both a one-argument method and a multi-arg method for both `Particles` and `StaticParticles`. If you only want to define one of these, see [`register_primitive_single`](@ref)/[`register_primitive_multi`](@ref). If the function is from base or stdlib, you can just add it to the appropriate list in the source and submit a PR :)

## Monte-Carlo simulation by `map/pmap`
Some functions will not work when the input arguments are of type `Particles`. For this kind of function, we provide a fallback onto a traditional `map(f,p.particles)`. The only thing you need to do is to decorate the function call with the function [`bymap`](@ref) or the macro [`@bymap`](@ref) like so:
```julia
f(x) = 3x^2
p = 1 ± 0.1
r = @bymap f(p) # bymap(f,p) may give better error traces
```
We further provide the macro [`@bypmap`](@ref) (and [`bypmap`](@ref)) which does exactly the same thing, but with a `pmap` (parallel map) instead, allowing you to run several invocations of `f` in a distributed fashion.

These utilities will map the function `f` over each element of `p::Particles{T,N}`, such that `f` is only called with arguments of type `T`, e.g., `Float64`. This handles arguments that are multivaiate particles `<: Vector{<:AbstractParticles}` as well.

These utilities will typically be slower than calling `f(p)`. If `f` is very expensive, [`@bypmap`](@ref) might prove prove faster than calling `f` with `p`, it's worth a try. The usual caveats for distributed computing applies, all code must be loaded on all workers etc.


## Array-to-array functions
These functions might not work with `Particles` out of the box. Special cases are currently implemented for
- `exp : ℝ(n×n) → ℝ(n×n)`   matrix exponential
- `log : ℝ(n×n) → C(n×n)`   matrix logarithm
- `eigvals : ℝ(n×n) → C(n)` **warning**: eigenvalues are sorted, when two eigenvalues cross, this function is nondifferentiable. Eigenvalues can thus appear to have dramatically widened distributions. Make sure you interpret the result of this call in the right way.

The function  [`ℝⁿ2ℝⁿ_function`](@ref)`(f::Function, p::AbstractArray{T})` applies `f : ℝⁿ → ℝⁿ` to an array of particles.

## Complex functions
These functions do not work with `Particles` out of the box. Special cases are currently implemented for
- `sqrt`, `exp`, `sin`, `cos`

We also provide in-place versions of the above functions, e.g.,
- `sqrt!(out, p)`, `exp!(out, p)`, `sin!(out, p)`, `cos!(out, p)`

The function [`ℂ2ℂ_function`](@ref)`(f::Function, z)` (`ℂ2ℂ_function!(f::Function, out, z)`) applies `f : ℂ → ℂ ` to `z::Complex{<:AbstractParticles}`.




## Difficult cases
Sometimes, defining a primitive function can be difficult, such as when the uncertain parameters are baked into some object. In such cases, we can call the function [`unsafe_comparisons`](@ref)`(true)`, which defines all comparison operators for uncertain values to compare using the `mean`. Note however that this enabling this is somewhat *unsafe* as this corresponds to a fallback to linear uncertainty propagation, why it's turned off by default. We also provide the macro
`@unsafe ex` to enable mean comparisons only locally in the expression `ex`.

In some cases, defining a primitive is not possible but allowing unsafe comparisons are not acceptable. One such case is functions that internally calculate eigenvalues of uncertain matrices. The eigenvalue calculation makes use of comparison operators. If the uncertainty is large, eigenvalues might change place in the sorted list of returned eigenvalues, completely ruining downstream computations. For this we recommend, in order of preference
1. Use [`bymap`](@ref). Applicable if all uncertain values appears as arguments to your entry function.
2. Create a [`Workspace`](@ref) object and call it using your entry function. Applicable if uncertain parameters appear nested in an object that is an argument to your entry function:
```julia
# desired computation: y = f(obj), obj contains uncertain parameters inside
y = with_workspace(f, obj)
# or equivalently
w = Workspace(f,obj) # This is somewhat expensive and can be reused
use_invokelatest = true # Set this to false to gain 0.1-1 ms, at the expense of world-age problems if w is created and used in the same function.
w(obj, use_invokelatest)
```
This interface is so far not tested very well and may throw strange errors. Some care has been taken to make error messages informative.
Internally, a `w::Workspace` object is created that tries to automatically construct an object identical to `obj`, but where all uncertain parameters are replaced by conventional `Real`. If the heuristics used fail, an error message is displayed detailing which method you need to implement to make it work. When called, `w` populates the internal buffer object with particle `i`, calls `f` using a `Particles`-free `obj` and stores the result in an output object at particle index  `i`. This is done for `i ∈ 1:N` after which the output is returned. Some caveats include: [`Workspace`](@ref) must not be created or used inside a `@generated` function.
