## Optimization =======================================

function perturb(rng::AbstractRNG, p, Cp)
    d = MvNormal(pmean(p), 1.1Cp + 1e-12I)
    Particles(rng, nparticles(p[1]), d)
end



"""
    res = optimize([rng::AbstractRNG,] f,p,τ=1,iters=10000)

Find the minimum of Function `f`, starting with initial distribution described by `p::Vector{Particles}`. `τ` is the initial temperature.
"""
function optimize(rng::AbstractRNG,f,p,τ=1; τi=1.005, iters=10000, tol=1e-8)
    p = deepcopy(p)
    N = nparticles(p[1])
    we = zeros(N)
    for i = 1:iters
        y  = -(f(p).particles)
        we .= exp.(τ.*y)
        j  = sample(rng,1:N, ProbabilityWeights(we), N, ordered=true)
        foreach(x->(x.particles .= x.particles[j]), p); # @test length(unique(p[1].particles)) == length(unique(j))
        Cp = pcov(p)
        tr(Cp) < tol && (@info "Converged at iteration $i"; return p)
        p = perturb(rng, p, Cp)
        τ *= τi
    end
    p
end
optimize(f,p,τ=1; kwargs...) = optimize(Random.GLOBAL_RNG,f,p,τ; kwargs...)
