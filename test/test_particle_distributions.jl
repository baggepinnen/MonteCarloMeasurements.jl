@testset "Particle Distributions" begin
    @info "Testing Particle Distributions"



pd = ParticleDistribution(Bernoulli, Particles(1000, Beta(2, 3)))
# @btime rand($pd) #  23.304 ns (0 allocations: 0 bytes)
# @btime rand(Bernoulli(0.3)) # 10.050 ns (0 allocations: 0 bytes)

@test pd[1] isa Bernoulli
@test length(pd) == 1
@test rand(pd) isa eltype(pd)
@test length(pd.d) == 1000

pd = ParticleDistribution(
    Normal,
    Particles(1000, Normal(10, 3)),
    Particles(1000, Normal(2, 0.1)),
)

@test pd[1] isa Normal
@test length(pd) == 1
@test rand(pd) isa eltype(pd)
@test length(pd.d) == 1000

@test_nowarn display(pd)


# @btime rand($pd) #  27.726 ns (0 allocations: 0 bytes)
# @btime rand(Normal(10,2)) # 12.788 ns (0 allocations: 0 bytes)


@unsafe d = Normal(Particles(),10+Particles())
@test d isa Normal{<:Particles{Float64}}
d2 = change_representation(Normal, d);
@test d2 isa Particles{Normal{Float64}}
d3 = change_representation(Normal, d2)
@test d3.μ == d.μ
@test d3.σ == d.σ



end
