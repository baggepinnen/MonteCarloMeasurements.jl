using MonteCarloMeasurements
using Test, LinearAlgebra, Statistics, Random
import MonteCarloMeasurements: ±, gradient

Random.seed!(0)

@testset "MonteCarloMeasurements.jl" begin

    # σ/√N = σm
    @testset "sampling" begin
        for _ = 1:10
            @test -3 < mean(systematic_sample(100))*sqrt(100) < 3
            @test -3 < mean(systematic_sample(10000))*sqrt(10000) < 3
            @test -0.9 < std(systematic_sample(100)) < 1.1
            @test -0.9 < std(systematic_sample(10000)) < 1.1
        end
        @test systematic_sample(10000, Normal(1,1)) |> Base.Fix1(fit, Normal) |> params |> x-> all(isapprox.(x,(1,1), atol=0.1))
        systematic_sample(10000, Gamma(1,1)) #|> Base.Fix1(fit, Gamma)
        systematic_sample(10000, TDist(1)) #|> Base.Fix1(fit, TDist)
        @test systematic_sample(10000, Beta(1,1)) |> Base.Fix1(fit, Beta) |> params |> x-> all(isapprox.(x,(1,1), atol=0.1))

    end
    @testset "Particles" begin
        for PT = (Particles, StaticParticles)

            p = PT(1000)
            @test 0 ± 1 ≈ p
            @test sum(p) ≈ 0
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
            @test_nowarn Particles(100, MvNormal(2,1)) ./ Particles(100, Normal(2,1))
        end
    end



    @testset "Multivariate Particles" begin
        for PT = (Particles, StaticParticles)

            p = PT(1000, MvNormal(2,1))
            @test_nowarn sum(p)
            @test cov(p) ≈ I atol=0.2
            @test mean(p) ≈ [0,0] atol=0.2
            @test size(Matrix(p)) == (1000,2)

            p = PT(100, MvNormal(2,2))
            @test cov(p) ≈ 4I atol=2
            @test mean(p) ≈ [0,0] atol=1
            @test size(Matrix(p)) == (100,2)

            p = PT(1000, MvNormal(2,2))
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
        # @test gradient(f,3) > 6 # Convex function
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
    @testset "leastsquares" begin
        n, m = 10000, 3
        A = randn(n,m)
        x = randn(m)
        y = A*x
        σ = 0.1
        yn = y .+ σ.*randn()
        # xh = A\y
        C1 = σ^2*inv(A'A)

        yp = y .+ σ.*Particles.(2000)
        xhp = (A'A)\A'yp
        @test sum(abs, tr((cov(xhp) .- C1) ./ abs.(C1))) < 0.2

        @test norm(cov(xhp) .- C1) < 1e-7
    end

    @testset "misc" begin
        p = 0 ± 1
        @test p[1] == p.particles[1]
        @test_nowarn display(p)
        @test_nowarn show(p)
        @test Particles{Float64,500}(p) == p
        @test length(Particles(100, MvNormal(2,1))) == 2
        @test length(p) == 500
        @test ndims(p) == 0
        @test eltype(p) == Float64
        @test Particles(500) + Particles(randn(Float32, 500)) isa typeof(Particles(500))
        @test_nowarn sqrt(complex(p,p)) == 1
        @test isfinite(p)
        @test round(p) ≈ 0 atol=0.1
        @test MvNormal(Particles(500, MvNormal(2,1))) isa MvNormal
        @test !(p<p)
        @test (p ≳ p)
        @test eps(typeof(p)) == eps(Float64)

        @test_nowarn plot(p)
        @test_nowarn errorbarplot(1:2,[p,p])
        @test_nowarn mcplot(1:2,[p,p])
        @test_nowarn ribbonplot(1:2,[p,p])

        @test_nowarn MonteCarloMeasurements.print_functions_to_extend()
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
