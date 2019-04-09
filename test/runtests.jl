using MonteCarloMeasurements
using Test, LinearAlgebra

# @testset "MonteCarloMeasurements.jl" begin
PT = Particles

f(x) = 2x + 10
p = PT(1000)
@test 9.6 < mean(f(p)) < 10.4
@test 9.6 < f(p) < 10.4
@test f(p) ≈ 10
@test !(f(p) ≲ 11)
@test f(p) ≲ 15
@test 5 ≲ f(p)
Normal(f(p)).μ ≈ mean(f(p))
@test cov(p) ≈ 1 atol=0.1
@test std(p) ≈ 1 atol=0.1
@test var(p) ≈ 1 atol=0.1

# plot(sin(0.1p)*sin(0.1p))

A = [PT(100) for i = 1:3, j = 1:3]
a = [PT(100) for i = 1:3]
b = [PT(100) for i = 1:3]
@test sum(a.*b) ≈ 0
@test all(A*b .≈ [0,0,0])

@test all(A\b .≈ zeros(3))
qr(A)


# using ControlSystems
# p = 1 + 0.1PT(100)
# G = tf([p], [1, 2, 1])

end

# using BenchmarkTools
# A = [StaticParticles(100) for i = 1:3, j = 1:3]
# B = similar(A, Float64)
# @btime qr($(copy(A)))
# @btime map(_->qr($B), 1:100);
