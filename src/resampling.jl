"""
    loglik = resample!(p::WeightedParticles)
Resample the particles based on the `p.logweights`. After a call to this function, weights will be reset to sum to one. Returns log-likelihood.
"""
function resample!(p::WeightedParticles)
    N = length(p)
    offset = maximum(p.logweights)
    p.logweights .= exp.(p.logweights .- offset)
    s = _resample!(p)
    # fill!(p.weights, 1/N)
    fill!(p.logweights, -log(N))
    log(s/exp(-offset)) - log(N)
end

"""
In-place systematic resampling of `p`, returns the sum of weights.
`p.logweights` should be exponentiated before calling this function.
"""
function _resample!(p::WeightedParticles)
    x,w = p.particles, p.logweights
    Σ = sum(w)
    N = length(w)
    bin = w[1]
    s = rand()*Σ/N
    bo = 1
    for i = 1:N
        @inbounds for b = bo:N
            if s < bin
                x[i] = x[b]
                bo = b
                break
            end
            bin += w[b+1] # should never reach here when b==N
        end
        s += Σ/N
    end
    Σ
end
