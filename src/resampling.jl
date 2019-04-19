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
"""
function _resample!(p::WeightedParticles)
    x,w = p.particles, p.logweights
    N = length(w)
    bin = w[1]
    s = rand()/N
    bo = 1
    for i = 1:N
        s += 1/N
        @inbounds for b = bo:N
            bin = bin + w[b]
            if s < bin
                x[i] = x[b]
                bo = b
                break
            end
        end
    end
    bin
end
