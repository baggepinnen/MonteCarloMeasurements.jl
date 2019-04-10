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

function jacobian(f::Function,p::MonteCarloMeasurements.MvParticles)
    r = (p-mean(p))\(f(p) - f(mean(p)))
end
