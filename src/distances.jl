"""
    wasserstein(p1::AbstractParticles,p2::AbstractParticles,p)

Returns the Wasserstein distance (Earth-movers distance) of order `p`, to the `p`th power, between `p1` and `p2`.
I.e., for `p=2`, this returns W₂²
"""
function wasserstein(p1::AbstractParticles,p2::AbstractParticles,p)
    p1 = sort(p1.particles)
    p2 = sort(p2.particles)
    wp = mean(eachindex(p1)) do i
        abs(p1[i]-p2[i])^p
    end
end
