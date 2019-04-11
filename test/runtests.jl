using MonteCarloMeasurements
using Test, LinearAlgebra, Statistics, Random
import MonteCarloMeasurements: ±, gradient
import Plots

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
            @test p <= p
            @test p >= p
            @test !(p ≲ p)
            @test !(p ≳ p)
            @test (p ≲ 2.1)
            @test !(p ≲ 1.9)
            @test (p ≳ -2.1)
            @test !(p ≳ -1.9)
            @test (-2.1 ≲ p)
            @test !(-1.9 ≲ p)
            @test (2.1 ≳ p)
            @test !(1.9 ≳ p)
            @test p ≈ p
            @test p ≈ 0
            @test 0 ≈ p
            @test p != 0
            @test p ≈ 1.9std(p)
            @test !(p ≈ 2.1std(p))
            @test p ≉ 2.1std(p)
            @test !(p ≉ 1.9std(p))


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
        @test_nowarn show(stdout, MIME"text/x-latex"(), p)
        @test Particles{Float64,500}(p) == p
        @test Particles{Float64,5}(0) == 0*Particles(5)
        @test length(Particles(100, MvNormal(2,1))) == 2
        @test length(p) == 500
        @test ndims(p) == 0
        @test eltype(typeof(p)) == Float64
        @test eltype(p) == Float64
        @test convert(Int, 0p) == 0
        @test promote_type(Particles{Float64,10}, Float64) == Particles{Float64,10}
        @test promote_type(Particles{Float64,10}, Int64) == Particles{Float64,10}
        @test promote_type(Particles{Float64,10}, ComplexF64) == Complex{Particles{Float64,10}}
        @test promote_type(Particles{Float64,10}, ComplexF64) == Complex{Particles{Float64,10}}
        @test convert(Float64, 0p) isa Float64
        @test convert(Float64, 0p) == 0
        @test convert(Int, 0p) isa Int
        @test convert(Int, 0p) == 0
        @test_throws ArgumentError convert(Int, p)
        @test_throws ArgumentError AbstractFloat(p)
        @test AbstractFloat(0p) == 0.0
        @test Particles(500) + Particles(randn(Float32, 500)) isa typeof(Particles(500))
        @test_nowarn sqrt(complex(p,p)) == 1
        @test isfinite(p)
        @test iszero(0p)
        @test !iszero(p)
        @test round(p) ≈ 0 atol=0.1
        @test norm(0p) == 0
        @test norm(p) ≈ 0 atol=0.01
        @test norm(p,Inf) > 0
        @test MvNormal(Particles(500, MvNormal(2,1))) isa MvNormal
        @test eps(typeof(p)) == eps(Float64)
        A = randn(2,2)
        B = A .± 0
        @test sum(abs, exp(A) .- exp(B)) < 1e-9
    end

    @testset "plotting" begin
        p = 0 ± 1
        v = [p,p]
        @test_nowarn Plots.plot(p)
        @test_nowarn Plots.plot(v)
        @test_nowarn Plots.plot(x->x^2,v)
        @test_nowarn Plots.plot(v,v)
        @test_nowarn Plots.plot(v,ones(2))
        @test_nowarn Plots.plot(1:2,v)

        @test_nowarn errorbarplot(1:2,v)
        @test_nowarn mcplot(1:2,v)
        @test_nowarn ribbonplot(1:2,v)

        @test_nowarn MonteCarloMeasurements.print_functions_to_extend()
    end
end


# Integration tests and bechmarks

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

## bode benchmark =========================================
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


## lsim etc. ==============================================
# using ControlSystems, MonteCarloMeasurements
# import MonteCarloMeasurements.±
# p = 1 ± 0.1
# ζ = 0.3 ± 0.1
# ω = 1 ± 0.1
# G = tf([p*ω], [1, 2ζ*ω, ω^2])
# # MonteCarloMeasurements.eval(:(Base.:(!=)(p1::AbstractParticles{T,N},p2::AbstractParticles{T,N}) where {T,N} = !(p1 ≈ p2))) # Hotpatch
# # Pd = c2d(G, 0.1)
#
# nyquist(G) .|> vec
# bode(G) .|> vec
# ss(G)
#
# y,t,x = step(Pd, 20)
# errorbarplot(t,y[:], 0.00, layout=3, subplot=1)
# errorbarplot!(t,y[:], 0.05, subplot=1)
# errorbarplot!(t,y[:], 0.1, subplot=1)
# mcplot!(t,y[:], subplot=2)
# ribbonplot!(t,y[:], subplot=3)

## Optim not working ===================================================
# using MonteCarloMeasurements, Optim
# p = 100.0 ± 10
# p2 = 1.0 ± 0.1
# rosenbrock(x) =  (p2 - x[1])^2 + p * (x[2] - x[1]^2)^2
# result = optimize(rosenbrock, zeros(2) .± 0, SimulatedAnnealing())
