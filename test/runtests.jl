using MonteCarloMeasurements
using Test, LinearAlgebra, Statistics
import MonteCarloMeasurements: ±, gradient

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
            @test 0 ± 1 ≈ p
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
            @test 0.9 < mean(f(p)) < 1.1
            @test 0.9 < mean(f(p)) < 1.1
            @test f(p) ≈ 1
            @test !(f(p) ≲ 1)
            @test f(p) ≲ 4
            @test -2.2 ≲ f(p)


            # plot(sin(0.1p)*sin(0.1p))

            A = randn(3,3) .+ [PT(100) for i = 1:3, j = 1:3]
            a = [PT(100) for i = 1:3]
            b = [PT(100) for i = 1:3]
            @test sum(a.*b) ≈ 0
            @test all(A*b .≈ [0,0,0])

            @test all(A\b .≈ zeros(3))
            @test_nowarn qr(A)
            @test_nowarn Particles(MvNormal(2,1)) ./ Particles(Normal(2,1))
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
    @testset "gradient" begin
        e = 0.001
        p = 3 ± e
        f = x -> x^2
        fp = f(p)
        @test gradient(f,p)[1] ≈ 6 atol=1e-4
        @test gradient(f,p)[2] ≈ 2e atol=1e-4
        @test gradient(f,3) > 6 # Convex function
        @test gradient(f,3) ≈ 6

        A = randn(3,3)
        H = A'A
        h = randn(3)
        c = randn()
        @assert isposdef(H)
        f = x -> (x'H*x + h'x) + c
        j = x -> H*x + h

        e = 0.001
        x = randn(3)
        xp = x ± e
        g = 2H*x + h
        @test MonteCarloMeasurements.gradient(f,xp) ≈ g atol = 0.1
        @test MonteCarloMeasurements.jacobian(j,xp) ≈ H
    end
end

# using BenchmarkTools
# A = [StaticParticles(100) for i = 1:3, j = 1:3]
# B = similar(A, Float64)
# @btime qr($(copy(A)))
# @btime map(_->qr($B), 1:100);

# using ControlSystems, MonteCarloMeasurements
# using Test, LinearAlgebra, Statistics
# import MonteCarloMeasurements: ±, gradient
# ##
# p = 1 ± 0.1
# ζ = 0.3 ± 0.1
# ω = 1 ± 0.1
# G = tf([p*ω], [1, 2ζ*ω, ω^2])
#
# dc = dcgain(G)
# w = exp10.(LinRange(-1,1,200))
# @time mag, phase = bode(G,w) .|> vec
#
# errorbarplot(w,mag, yscale=:log10, xscale=:log10)
# mcplot(w,mag, yscale=:log10, xscale=:log10, alpha=0.2)
# ribbonplot(w,mag, yscale=:identity, xscale=:log10, alpha=0.2)

# using BenchmarkTools, Printf
# p = 1 ± 0.1
# ζ = 0.3 ± 0.1
# ω = 1 ± 0.1
# G = tf([p*ω], [1, 2ζ*ω, ω^2])
# t1 = @belapsed bode($G,$w)
# p = 1
# ζ = 0.3
# ω = 1
# G = tf([p*ω], [1, 2ζ*ω, ω^2])
# t2 = @belapsed bode($G,$w)
#
# @printf("Time with 500 particles: %16.4fms \nTime with regular floating point: %7.4fms\n500×floating point time: %16.4fms\nSpeedup factor: %22.1fx\n", 1000*t1, 1000*t2, 1000*500t2, 500t2/t1)
