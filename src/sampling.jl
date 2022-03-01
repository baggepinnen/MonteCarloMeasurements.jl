"""
    systematic_sample([rng::AbstractRNG,] N, d=Normal(0,1); permute=true)

returns a `Vector` of length `N` sampled systematically from the distribution `d`. If `permute=false`, this vector will be sorted.
"""
function systematic_sample(rng::AbstractRNG, N, d=Normal(0,1); permute=true)
    T = eltype(mean(d)) # eltype(d) does not appear to be sufficient
    e = T(0.5/N) # rand()/N
    y = e:1/N:1
    o = quantile.((d, ),y)
    permute && permute!(o, randperm(rng, N))
    return eltype(o) == T ? o : T.(o)
end

function systematic_sample(N, d=Normal(0,1); kwargs...)
    return systematic_sample(Random.GLOBAL_RNG, N, d; kwargs...)
end

"""
    ess(p::AbstractParticles{T,N})

Calculates the effective sample size. This is useful if particles come from MCMC sampling and are correlated in time. The ESS is a number between [0,N].

Initial source: https://github.com/tpapp/MCMCDiagnostics.jl
"""
function ess(p::AbstractParticles)
    ac    = autocor(p.particles,1:min(250, nparticles(p)÷2))
    N     = length(ac)
    τ_inv = 1 + 2ac[1]
    K     = 2
    while K < N - 2
        Δ = ac[K] + ac[K+1]
        Δ < 0 && break
        τ_inv += 2Δ
        K += 2
    end
    min(1 / τ_inv, one(τ_inv))*nparticles(p)
end
