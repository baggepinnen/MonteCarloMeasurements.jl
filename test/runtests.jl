using MonteCarloMeasurements
using Test, LinearAlgebra, Statistics

@testset "MonteCarloMeasurements.jl" begin

    # σ/√N = σm
    @testset "sampling" begin
        for _ = 1:10
            @test -3 < mean(sysrandn(100))*sqrt(100) < 3
            @test -3 < mean(sysrandn(10000))*sqrt(10000) < 3
            @test -0.9 < std(sysrandn(100)) < 1.1
            @test -0.9 < std(sysrandn(10000)) < 1.1
        end
    end
    @testset "Particles" begin
        for PT = (Particles, StaticParticles)

            p = PT(1000)
            @test cov(p) ≈ 1 atol=0.2
            @test std(p) ≈ 1 atol=0.2
            @test var(p) ≈ 1 atol=0.2

            f = x -> 2x + 10
            @test 9.6 < mean(f(p)) < 10.4
            @test 9.6 < f(p) < 10.4
            @test f(p) ≈ 10
            @test !(f(p) ≲ 11)
            @test f(p) ≲ 15
            @test 5 ≲ f(p)
            @test Normal(f(p)).μ ≈ mean(f(p))
            @test fit(Normal, f(p)).μ ≈ mean(f(p))

            f = x -> x^2
            p = PT(1000)
            @test -0.4 < mean(f(p)) < 0.4
            @test -0.4 < f(p) < 10.4
            @test f(p) ≈ 0
            @test !(f(p) ≲ 1)
            @test f(p) ≲ 2.2
            @test -2.2 ≲ f(p)


            # plot(sin(0.1p)*sin(0.1p))

            A = randn(3,3) .+ [PT(100) for i = 1:3, j = 1:3]
            a = [PT(100) for i = 1:3]
            b = [PT(100) for i = 1:3]
            @test sum(a.*b) ≈ 0
            @test all(A*b .≈ [0,0,0])

            @test all(A\b .≈ zeros(3))
            @test_nowarn qr(A)

        end
    end



    @testset "Multivariate Particles" begin
        for PT = (Particles, StaticParticles)

            p = PT(MvNormal(2,1), 1000)
            @test cov(p) ≈ I atol=0.2
            @test mean(p) ≈ [0,0] atol=0.1
            @test size(Matrix(p)) == (1000,2)

            p = PT(MvNormal(2,2), 100)
            @test cov(p) ≈ 4I atol=2
            @test mean(p) ≈ [0,0] atol=1
            @test size(Matrix(p)) == (100,2)

            p = PT(MvNormal(2,2), 1000)
            @test fit(MvNormal, p).μ ≈ mean(p)
            @test MvNormal(p).μ ≈ mean(p)
            @test cov(MvNormal(p)) ≈ cov(p)
        end
    end
end

# using BenchmarkTools
# A = [StaticParticles(100) for i = 1:3, j = 1:3]
# B = similar(A, Float64)
# @btime qr($(copy(A)))
# @btime map(_->qr($B), 1:100);

# using ControlSystems
# p = 1 + 0.1PT(100)
# G = tf([p], [1, 2, 1])
