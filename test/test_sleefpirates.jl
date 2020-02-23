@testset "SLEEFPirates" begin
    @info "Testing SLEEFPirates"

    using SLEEFPirates
    a = rand(500)
    @test map(exp, a) ≈ exp(Particles(a)).particles
    @test map(log, a) ≈ log(Particles(a)).particles
    @test map(cos, a) ≈ cos(Particles(a)).particles

    @test map(exp, a) ≈ exp(StaticParticles(a)).particles
    @test map(log, a) ≈ log(StaticParticles(a)).particles
    @test map(cos, a) ≈ cos(StaticParticles(a)).particles


    a = rand(Float32, 500)
    @test map(exp, a) ≈ exp(Particles(a)).particles
    @test map(log, a) ≈ log(Particles(a)).particles
    @test map(cos, a) ≈ cos(Particles(a)).particles

    @test map(exp, a) ≈ exp(StaticParticles(a)).particles
    @test map(log, a) ≈ log(StaticParticles(a)).particles
    @test map(cos, a) ≈ cos(StaticParticles(a)).particles

end
