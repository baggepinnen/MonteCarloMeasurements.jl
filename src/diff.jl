"""
    gradient(f, p::AbstractParticles)

Calculate the gradient of `f` in `p`. This corresponds to a smoothed finite-difference approximation where the smoothing kernel is given by the distribution of `p`.
Return mean and std.
"""
function gradient(f,p::MonteCarloMeasurements.AbstractParticles)
    r = 2(p\f(p))
    mean(r), std(r)
end
function gradient(f,p::Union{Integer, AbstractFloat})
    p = p  ± 0.000001
    r = 2(p\f(p))
    mean(r)
end

function gradient(f::Function,p::MonteCarloMeasurements.MvParticles)
    r = (p-mean(p))\(f(p) - f(mean(p)))
end

"""
    jacobian(f::Function, p::Vector{<:AbstractParticles})

Calculate the Jacobian of `f` in `p`. This corresponds to a smoothed finite-difference approximation where the smoothing kernel is given by the distribution of `p`.
"""
function jacobian(f::Function,p::MonteCarloMeasurements.MvParticles)
    r = (p-mean(p))\(f(p) - f(mean(p)))
end
