using MonteCarloMeasurements
using Test

# @testset "MonteCarloMeasurements.jl" begin

f(x) = 2x + 10
p = Particles(10000)
@test 9.96 < mean(f(p)) < 10.04
@test 9.96 < f(p) < 10.04
@test f(p) ≈ 10
@test !(f(p) ≲ 11)
@test f(p) ≲ 15
@test 5 ≲ f(p)
Normal(f(p)).μ ≈ mean(f(p))

# plot(sin(0.1p)*sin(0.1p))

A = [Particles(1000) for i = 1:3, j = 1:3]
a = [Particles(1000) for i = 1:3]
b = [Particles(1000) for i = 1:3]
@test sum(a.*b) ≈ 0
@test all(A*b .≈ [0,0,0])



using ControlSystems
p = 1 + 0.1Particles(100)
G = tf([p], [1, 2, 1])

# end
