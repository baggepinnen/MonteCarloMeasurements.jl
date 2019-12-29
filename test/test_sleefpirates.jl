using SLEEFPirates
a = rand(500)
@test map(exp, a) ≈ exp(Particles(a)).particles
@test map(log, a) ≈ log(Particles(a)).particles
@test map(cos, a) ≈ cos(Particles(a)).particles


a = rand(Float32, 500)
@test map(exp, a) ≈ exp(Particles(a)).particles
@test map(log, a) ≈ log(Particles(a)).particles
@test map(cos, a) ≈ cos(Particles(a)).particles
