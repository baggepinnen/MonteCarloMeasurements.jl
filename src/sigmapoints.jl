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
"""
function sigmapoints(m, Σ)
    n = length(m)
    X = sqrt(n*Σ)
    [X; -X; zeros(1,n)] .+ m'
end

sigmapoints(d::Normal) = sigmapoints(mean(d), var(d))
sigmapoints(d::MvNormal) = sigmapoints(mean(d), Matrix(cov(d)))


"""
    Y = transform_moments(X::Matrix, m, Σ)
Transforms `X` such that it get the specified mean and covariance.

```julia
m, Σ   = [1,2], [2 1; 1 4] # Desired mean and covariance
particles = transform_moments(X, m, Σ)
julia> cov(particles) ≈ Σ
true
```
**Note**, if `X` is a latin hypercube and `Σ` is non-diagonal, then the latin property is destroyed for all dimensions but the first.
"""
function transform_moments(X,m,Σ)
    X  = X .- mean(X,dims=1) # Normalize the sample
    xl = cholesky(cov(X)).L
    Matrix((m .+ (cholesky(Σ).L/xl)*X')')
end
