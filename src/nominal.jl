function shuffle_and_insert(p::AbstractParticles, ind, val)
    p = deepcopy(p)
    p.particles[ind] = p.particles[1]
    p.particles[1] = val
    p
end

function shuffle_and_insert(p::StaticParticles, ind, val)
    part = p.particles
    part = setindex(part, part[1], ind)
    part = setindex(part, val, 1)
    StaticParticles(part)
end

"""
    pn = with_nominal(p, val)

Endow particles `p` with a nominal value `val`. The particle closest to `val` will be replaced with val, and moved to index 1. This operation introduces a slight bias in the statistics of `pn`, but the operation is asymptotically unbiased for large sample sizes. To obtain the nominal value of `pn`, call `nominal(pn)`.
"""
function with_nominal(p::AbstractParticles, val)
    minind = argmin(abs.(p.particles .- val))
    shuffle_and_insert(p, minind, val)
end

function with_nominal(p::MvParticles, val::AbstractVector)
    M = Matrix(p)
    minind = argmin(vec(sum(abs2, M .- val', dims=1)))
    shuffle_and_insert.(p, minind, val)
end

"""
    nominal(p)

Return the nominal value of `p` (assumes that `p` has been endowed with a nominal value using `with_nominal`).
"""
nominal(p::AbstractParticles) = p.particles[1]
nominal(p::MvParticles) = nominal.(p)
nominal(P) = replace_particles(P, replacer=nominal)
