# """
#     logΣexp, Σexp = logsumexp!(p::WeightedParticles)
# Return log(∑exp(w)). Modifies the weight vector to `w = exp(w-offset)`
# Uses a numerically stable algorithm with offset to control for overflow and `log1p` to control for underflow. `Σexp` is the sum of the weifhts in the state they are left, i.e., `sum(exp.(w).-offset)`.
#
# References:
# https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
# https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
# """
# function logsumexp!(p::WeightedParticles)
#     N = length(p)
#     w = p.logweights
#     offset, maxind = findmax(w)
#     w .= exp.(w .- offset)
#     Σ = sum_all_but(w,maxind) # Σ = ∑wₑ-1
#     log1p(Σ) + offset, Σ+1
# end
#
# """
#     sum_all_but(w, i)
#
# Add all elements of vector `w` except for index `i`. The element at index `i` is assumed to have value 1
# """
# function sum_all_but(w,i)
#     w[i] -= 1
#     s = sum(w)
#     w[i] += 1
#     s
# end
#
# """
#     loglik = resample!(p::WeightedParticles)
# Resample the particles based on the `p.logweights`. After a call to this function, weights will be reset to sum to one. Returns log-likelihood.
# """
# function resample!(p::WeightedParticles)
#     N = length(p)
#     w = p.logweights
#     logΣexp,Σ = logsumexp!(p)
#     _resample!(p,Σ)
#     # fill!(p.weights, 1/N)
#     fill!(w, -log(N))
#     logΣexp - log(N)
# end

# """
# In-place systematic resampling of `p`, returns the sum of weights.
# `p.logweights` should be exponentiated before calling this function.
# """
# function _resample!(p::WeightedParticles,Σ)
#     x,w = p.particles, p.logweights
#     N = length(w)
#     bin = w[1]
#     s = rand()*Σ/N
#     bo = 1
#     for i = 1:N
#         @inbounds for b = bo:N
#             if s < bin
#                 x[i] = x[b]
#                 bo = b
#                 break
#             end
#             bin += w[b+1] # should never reach here when b==N
#         end
#         s += Σ/N
#     end
#     Σ
# end

for PT in ParticleSymbols
    @eval begin
        """
            bootstrap([rng::AbstractRNG,] p::Particles, n = nparticles(p))

        Return Particles resampled with replacement. `n` specifies the number of samples to draw. Also works for arrays of Particles, in which case a single set of indices are drawn and used to extract samples from all elements in the array.
        """
        function bootstrap(rng::AbstractRNG, p::$PT, n::Integer = nparticles(p))
            $PT(p.particles[[rand(rng, 1:nparticles(p)) for _ in 1:n]])
        end

        function bootstrap(rng::AbstractRNG, p::AbstractArray{<:$PT}, n::Integer = nparticles(p))
            inds = [rand(rng, 1:nparticles(p)) for _ in 1:n]
            newpart = [p.particles[inds] for p in p]
            $PT.(newpart)
        end
    end
end
bootstrap(p::T, n::Integer = nparticles(p)) where T <: AbstractParticles = bootstrap(Random.GLOBAL_RNG, p, n
bootstrap(p::MvParticles, n::Integer = nparticles(p)) = bootstrap(Random.GLOBAL_RNG, p, n)
