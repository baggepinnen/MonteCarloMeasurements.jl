"""
    sigmapoints(m, Σ)
    sigmapoints(d::Normal)
    sigmapoints(d::MvNormal)

The [unscented transform](https://en.wikipedia.org/wiki/Unscented_transform#Sigma_points) uses a small number of points to propagate the first and second moments of a probability density, called *sigma points*. We provide a function `sigmapoints(μ, Σ)` that creates a `Matrix` of `2n+1` sigma points, where `n` is the dimension. This can be used to initialize any kind of `AbstractParticles`, e.g.:
```julia
julia> m = [1,2]

julia> Σ = [3. 1; 1 4]

julia> p = StaticParticles(sigmapoints(m,Σ))
2-element Array{StaticParticles{Float64,5},1}:
 (5 StaticParticles: 1.0 ± 1.73)
 (5 StaticParticles: 2.0 ± 2.0)

julia> cov(p) ≈ Σ
true

julia> mean(p) ≈ m
true
```
Make sure to pass the variance (not std) as second argument in case `μ` and `Σ` are scalars.

# Caveat
If you are creating several one-dimensional uncertain values using sigmapoints independently, they will be strongly correlated. Use the multidimensional constructor! Example:
```julia
p = StaticParticles(sigmapoints(1, 0.1^2))               # Wrong!
ζ = StaticParticles(sigmapoints(0.3, 0.1^2))             # Wrong!
ω = StaticParticles(sigmapoints(1, 0.1^2))               # Wrong!

p,ζ,ω = StaticParticles(sigmapoints([1, 0.3, 1], 0.1^2)) # Correct
```
"""
function sigmapoints(m, Σ::AbstractMatrix)
    n = length(m)
    # X = sqrt(n*Σ)
    X = cholesky(Symmetric(n*Σ)).U # Much faster than sqrt
    T = promote_type(eltype(m), eltype(X))
    [X; -X; zeros(T,1,n)] .+ m'
end

sigmapoints(m, Σ::Number) = sigmapoints(m, diagm(0=>fill(Σ, length(m))))
sigmapoints(d::Normal) = sigmapoints(mean(d), var(d))
sigmapoints(d::MvNormal) = sigmapoints(mean(d), Matrix(cov(d)))


"""
    Y = transform_moments(X::Matrix, m, Σ; preserve_latin=false)
Transforms `X` such that it get the specified mean and covariance.

```julia
m, Σ   = [1,2], [2 1; 1 4] # Desired mean and covariance
particles = transform_moments(X, m, Σ)
julia> cov(particles) ≈ Σ
true
```
**Note**, if `X` is a latin hypercube and `Σ` is non-diagonal, then the latin property is destroyed for all dimensions but the first.
We provide a method `preserve_latin=true`) which absolutely preserves the latin property in all dimensions, but if you use this, the covariance of the sample will be slightly wrong
"""
function transform_moments(X,m,Σ; preserve_latin=false)
    X  = X .- mean(X,dims=1) # Normalize the sample
    if preserve_latin
        xl = Diagonal(std(X,dims=1)[:])
        # xl = cholesky(Diagonal(var(X,dims=1)[:])).L
    else
        xl = cholesky(cov(X)).L
    end
    Matrix((m .+ (cholesky(Σ).L/xl)*X')')
end
