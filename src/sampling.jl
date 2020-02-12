"""
    systematic_sample([rng::AbstractRNG,] N, d=Normal(0,1); permute=true)

returns a `Vector` of length `N` sampled systematically from the distribution `d`. If `permute=false`, this vector will be sorted.
"""
function systematic_sample(rng::AbstractRNG, N, d=Normal(0,1); permute=true)
    T = eltype(d)
    e = T(0.5/N) # rand()/N
    y = e:1/N:1
    o = map(y) do y
        quantile(d,y)
    end
    permute && permute!(o, randperm(rng, N))
    return eltype(o) == T ? o : T.(o)
end
function systematic_sample(N, d=Normal(0,1); kwargs...)
    return systematic_sample(Random.GLOBAL_RNG, N, d; kwargs...)
end
