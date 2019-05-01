## Optimization =======================================

function perturb(p, Cp)
    d = MvNormal(mean(p), 1.1Cp + 1e-16I)
    Particles(length(p[1]), d)
end



"""
    res = optimize(f,p,τ=1)

Find the minimum of Function `f`, starting with initial distribution described by `p::Vector{Particles}`. `τ` is the initial temperature.
"""
function optimize(f,p,τ=1)
    p = deepcopy(p)
    N = length(p[1])
    we = zeros(N)
    for i = 1:10000
        y  = -(f(p).particles)
        we .= exp.(τ.*y)
        j  = sample(1:N, ProbabilityWeights(we), N, ordered=true)
        foreach(x->(x.particles .= x.particles[j]), p); # @test length(unique(p[1].particles)) == length(unique(j))
        Cp = cov(p)
        tr(Cp) < 1e-8 && (@info "Converged at iteration $i"; return p)
        p = perturb(p, Cp)
        τ *= 1.005
    end
    p
end
