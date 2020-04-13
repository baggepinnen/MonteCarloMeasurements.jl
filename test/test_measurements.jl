
@testset "Measurements" begin
    @info "Testing Measurements"
    import Measurements

    m = Measurements.measurement(1,2)
    @test Particles(m) isa Particles{Float64, MonteCarloMeasurements.DEFAULT_NUM_PARTICLES}
    @test StaticParticles(m) isa StaticParticles{Float64, MonteCarloMeasurements.DEFAULT_STATIC_NUM_PARTICLES}


    @test Particles(10, m) isa Particles{Float64, 10}
    @test StaticParticles(10, m) isa StaticParticles{Float64, 10}



    @test Particles.([m, m]) isa Vector{Particles{Float64, MonteCarloMeasurements.DEFAULT_NUM_PARTICLES}}
    @test StaticParticles.([m, m]) isa Vector{StaticParticles{Float64, MonteCarloMeasurements.DEFAULT_STATIC_NUM_PARTICLES}}

    @test Particles(m) ≈ 1 ± 2
    @test std(Particles(m)) ≈ 2 atol=1e-3
    @test mean(Particles(m)) ≈ 1 atol=1e-3

    @test Measurements.uncertainty(Particles(m)) ≈ 2 atol=1e-3
    @test Measurements.value(Particles(m)) ≈ 1 atol=1e-3

end
