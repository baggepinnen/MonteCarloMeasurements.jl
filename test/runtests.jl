@info "Running tests"
using MonteCarloMeasurements, Distributions
using Test, LinearAlgebra, Statistics, Random
import MonteCarloMeasurements: ⊗, gradient, optimize, DEFAULT_NUM_PARTICLES
@info "import Plots"
import Plots
@info "import Plots done"

Random.seed!(0)

@testset "MonteCarloMeasurements.jl" begin
    @info "Testing MonteCarloMeasurements"

    # σ/√N = σm
    @time @testset "sampling" begin
        @info "Testing sampling"
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
    @time @testset "Basic operations" begin
        @info "Creating the first StaticParticles"
        @test 0 ∓ 1 isa StaticParticles
        @test [0,0] ∓ 1. isa MonteCarloMeasurements.MvParticles
        @test [0,0] ∓ [1.,1.] isa MonteCarloMeasurements.MvParticles

        @info "Done"
        PT = Particles
        for PT = (Particles, StaticParticles)
            @testset "$(repr(PT))" begin
                @info "Running tests for $PT"
                p = PT(100)
                @test_nowarn MonteCarloMeasurements.shortform(p)
                @test_nowarn println(p)
                @test (p+p+p).particles ≈ 3p.particles # Test 3arg operator
                @test (p+p+1).particles ≈ 1 .+ 2p.particles # Test 3arg operator
                @test (1+p+1).particles ≈ 2 .+ p.particles # Test 3arg operator
                @test (p+1+p).particles ≈ 1 .+ 2p.particles # Test 3arg operator
                @test 0 ± 1 ≈ p
                @test 0 ∓ 1 ≈ p
                @test sum(p) ≈ 0
                @test cov(p) ≈ 1 atol=0.2
                @test std(p) ≈ 1 atol=0.2
                @test var(p) ≈ 1 atol=0.2
                @test meanvar(p) ≈ 1/(length(p)) atol=5e-3
                @test meanstd(p) ≈ 1/sqrt(length(p)) atol=5e-3
                @test minmax(1+p,p) == (p, 1+p)

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
                @test p != 2p
                @test p ≈ 1.9std(p)
                @test !(p ≈ 2.1std(p))
                @test !(p ≉ p)
                @test !(mean(p) ≉ p)
                @test p ≉ 2.1std(p)
                @test !(p ≉ 1.9std(p))
                @test (5 ± 0.1) ≳ (1 ± 0.1)
                @test (1 ± 0.1) ≲ (5 ± 0.1)

                v = randn(5)
                @test Vector(PT(v)) == v
                @test Array(PT(v)) == v

                @testset "unsafe comparisons" begin
                    unsafe_comparisons(false)
                    @test_throws ErrorException p<p
                    @test_throws ErrorException p>p
                    @test_throws ErrorException p>=p
                    @test_throws ErrorException p<=p

                    for mode in (:montecarlo, :reduction, :safe)
                        @show mode
                        unsafe_comparisons(mode, verbose=false)
                        @test p<100+p
                        @test p+100>p
                        @test p+100>=p
                        @test p<=100+p

                        @test p<100
                        @test 100>p
                        @test 100>=p
                        @test p<=100
                        @unsafe begin
                            @test -10 < p
                            @test p <= p
                            @test p >= p
                            @test !(p < p)
                            @test !(p > p)
                            @test (p < 1+p)
                            @test (p+1 > p)
                        end
                    end

                    @test_throws ErrorException p<p
                    @test_throws ErrorException p>p
                    @test_throws ErrorException @unsafe error("") # Should still be safe after error

                    @test_throws ErrorException p>=p
                    @test_throws ErrorException p<=p
                    unsafe_comparisons(:montecarlo, verbose=false)
                    @test p>=p
                    @test p<=p
                    @test !(p<p)
                    @test !(p>p)
                    @test_throws ErrorException p < Particles(shuffle(p.particles))
                    unsafe_comparisons(false)

                    @unsafe tv = 2
                    @test tv == 2
                    @unsafe tv1,tv2 = 1,2
                    @test (tv1,tv2) == (1,2)
                    @unsafe tv3,tv4 = range(1, stop=3, length=5), range(1, stop=3, length=5)
                    @test (tv3,tv4) == (range(1, stop=3, length=5),range(1, stop=3, length=5))
                    @test MonteCarloMeasurements.COMPARISON_FUNCTION[] == mean
                    set_comparison_function(median)
                    @test MonteCarloMeasurements.COMPARISON_FUNCTION[] == median
                    cp = PT(10)
                    cmp = @unsafe p < cp
                    @test cmp == (median(p) < median(cp))
                    set_comparison_function(mean)
                end


                f = x -> 2x + 10
                @test 9.6 < mean(f(p)) < 10.4
                # @test 9.6 < f(p) < 10.4
                @test f(p) ≈ 10
                @test !(f(p) ≲ 11)
                @test f(p) ≲ 15
                @test 5 ≲ f(p)
                @test 1 ≲ 2

                @test Normal(f(p)).μ ≈ mean(f(p))
                @test fit(Normal, f(p)).μ ≈ mean(f(p))


                f = x -> x^2
                p = PT(100)
                @test 0.9 < mean(f(p)) < 1.1
                @test 0.9 < mean(f(p)) < 1.1
                @test f(p) ≈ 1
                @test !(f(p) ≲ 1)
                @test f(p) ≲ 5
                @test -3 ≲ f(p)
                @test MvNormal([f(p),p]) isa MvNormal

                A = randn(3,3) .+ [PT(100) for i = 1:3, j = 1:3]
                a = [PT(100) for i = 1:3]
                b = [PT(100) for i = 1:3]
                @test sum(a.*b) ≈ 0
                @test all(A*b .≈ [0,0,0])
                @test A*b .+ 1 ≈ [1,1,1]

                @test all(A\b .≈ zeros(3))
                @test_nowarn @unsafe qr(A)
                @test_nowarn Particles(100, MvNormal(2,1)) ./ Particles(100, Normal(2,1))
                pn = Particles(100, Normal(2,1), systematic=false)
                @test pn ≈ 2
                @test !issorted(pn.particles)
                @test !issorted(p.particles)

                pn = Particles(100, Normal(2,1), systematic=true, permute=false)
                @test pn ≈ 2
                @test issorted(pn.particles)

                rng = MersenneTwister(657)
                pn1 = Particles(rng, 100, Normal(2,1), systematic=true, permute=true)
                rng = MersenneTwister(657)
                pn2 = Particles(rng, 100, Normal(2,1), systematic=true, permute=true)
                @test pn1 == pn2

                rng = MersenneTwister(27)
                pn1 = Particles(rng, Normal(2,1), systematic=true, permute=true)
                rng = MersenneTwister(27)
                pn2 = Particles(rng, Normal(2,1), systematic=true, permute=true)
                @test pn1 == pn2

                rng = MersenneTwister(932)
                pn1 = Particles(rng, 100, systematic=true, permute=true)
                rng = MersenneTwister(932)
                pn2 = Particles(rng, 100, systematic=true, permute=true)
                @test pn1 == pn2

                @info "Tests for $PT done"

                p = PT{Float64,10}(2)
                @test p isa PT{Float64,10}
                @test all(p.particles .== 2)

                @test Particles(100) + Particles(randn(Float32, 100)) ≈ 0
                @test_throws MethodError p + Particles(randn(Float32, 200)) # Npart and Float type differ
                @test_throws MethodError p + Particles(200) # Npart differ

                mp = MvParticles([randn(10) for _ in 1:3])
                @test length(mp) == 10
                @test MonteCarloMeasurements.nparticles(mp) == 3

                @testset "discrete distributions" begin
                    p = PT(Poisson(50))
                    @test p isa PT{Int}
                    @test (p^2 - 1) isa PT{Int}
                    @test exp(p) isa PT{Float64}

                    # mainly just test that printing PT{Int} doesn't error
                    io = IOBuffer()
                    show(io, p)
                    s = String(take!(io))
                    @test occursin('±', s)

                    # issue #50
                    @test 2.5 * p isa PT{Float64}
                    @test p / 3 isa PT{Float64}
                    @test sqrt(p) isa PT{Float64}
                end
            end
        end
    end



    @time @testset "Multivariate Particles" begin
        for PT = (Particles, StaticParticles)
            @testset "$(repr(PT))" begin
                @info "Running tests for multivariate $PT"
                p = PT(100, MvNormal(2,1))
                @test_nowarn sum(p)
                @test cov(p) ≈ I atol=0.6
                @test cor(p) ≈ I atol=0.6
                @test mean(p) ≈ [0,0] atol=0.2
                m = Matrix(p)
                @test size(m) == (100,2)
                # @test m[1,2] == p[1,2]

                p = PT(100, MvNormal(2,2))
                @test cov(p) ≈ 4I atol=2
                @test [0,0] ≈ mean(p) atol=1
                @test size(Matrix(p)) == (100,2)

                p = PT(100, MvNormal(2,2))
                @test fit(MvNormal, p).μ ≈ mean(p)
                @test MvNormal(p).μ ≈ mean(p)
                @test cov(MvNormal(p)) ≈ cov(p)
                @info "Tests for multivariate $PT done"
            end
        end
    end

    @testset "sigmapoints" begin
        @info "Testing sigmapoints"
        m = 1
        Σ = 3
        s = sigmapoints(m,Σ)
        @test var(s) ≈ Σ
        @test mean(s) == m
        @test sigmapoints(Normal(m,√(Σ))) == s

        m = [1,2]
        Σ = [3. 1; 1 4]
        s = sigmapoints(m,Σ)
        @test cov(s) ≈ Σ
        @test mean(s, dims=1)' ≈ m
        @test sigmapoints(MvNormal(m,Σ)) == s
    end

    @testset "transform_moments" begin
        m, Σ   = [1,2], [2 1; 1 4] # Desired mean and covariance
        C = randn(2,2)
        C = cholesky(C'C + 5I).L
        particles = transform_moments((C*randn(2,DEFAULT_NUM_PARTICLES))', m, Σ)
        @test mean(particles, dims=1)[:] ≈ m
        @test cov(particles) ≈ Σ
        particles = transform_moments((C*randn(2,DEFAULT_NUM_PARTICLES))', m, Σ, preserve_latin=true)
        @test mean(particles, dims=1)[:] ≈ m
        @test Diagonal(cov(particles)) ≈ Diagonal(Σ) atol=2
    end

    @time @testset "gradient" begin
        @info "Testing gradient"
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
    @time @testset "leastsquares" begin
        @info "Testing leastsquares"
        n, m = 10000, 3
        A = randn(n,m)
        x = randn(m)
        y = A*x
        σ = 0.1
        yn = y .+ σ.*randn()
        # xh = A\y
        C1 = σ^2*inv(A'A)

        yp = yn .+ σ.*Particles.(2000)
        xhp = (A'A)\(A'yp)
        @test xhp ≈ A\yp
        @test sum(abs, tr((cov(xhp) .- C1) ./ abs.(C1))) < 0.2

        @test norm(cov(xhp) .- C1) < 1e-7
        @test xhp ≈ x
        @test mean(xhp) ≈ x atol=3sum(sqrt.(diag(C1)))
        yp = nothing; GC.gc(true) # make sure this big matrix is deallocated
    end

    @time @testset "misc" begin
        @info "Testing misc"
        p = 0 ± 1
        @test p[1] == p.particles[1]
        @test MonteCarloMeasurements.particletypetuple(p) == (Float64, DEFAULT_NUM_PARTICLES, Particles)
        @test MonteCarloMeasurements.particletypetuple(typeof(p)) == (Float64, DEFAULT_NUM_PARTICLES, Particles)
        @test_nowarn display(p)
        @test_nowarn println(p)
        @test_nowarn show(stdout, MIME"text/x-latex"(), p); println()
        @test_nowarn println(0p)
        @test_nowarn show(stdout, MIME"text/x-latex"(), 0p); println()
        @test_nowarn show(stdout, MIME"text/x-latex"(), p + im*p); println()
        @test_nowarn show(stdout, MIME"text/x-latex"(), p - im*p); println()
        @test_nowarn show(stdout, MIME"text/x-latex"(), im*p); println()
        @test_nowarn show(stdout, MIME"text/x-latex"(), -im*p); println()

        @test_nowarn show(stdout, MIME"text/plain"(), p); println()
        @test_nowarn println(0p)
        @test_nowarn show(stdout, MIME"text/plain"(), 0p); println()
        @test_nowarn show(stdout, MIME"text/plain"(), p + im*p); println()
        @test_nowarn show(stdout, MIME"text/plain"(), p - im*p); println()
        @test_nowarn show(stdout, MIME"text/plain"(), im*p); println()
        @test_nowarn show(stdout, MIME"text/plain"(), -im*p); println()

        @test_nowarn display([p, p])
        @test_nowarn println([p, p])
        @test_nowarn println([p, 0p])

        @test Particles{Float64,DEFAULT_NUM_PARTICLES}(p) == p
        @test Particles{Float64,5}(0) == 0*Particles(5)
        @test length(Particles(100, MvNormal(2,1))) == 2
        @test length(p) == DEFAULT_NUM_PARTICLES
        @test ndims(p) == 0
        @test eltype(typeof(p)) == typeof(p)
        @test eltype(p) == typeof(p)
        @test convert(Int, 0p) == 0
        @test promote_type(Particles{Float64,10}, Float64) == Particles{Float64,10}
        @test promote_type(Particles{Float64,10}, Int64) == Particles{Float64,10}
        @test promote_type(Particles{Float64,10}, ComplexF64) == Complex{Particles{Float64,10}}
        @test promote_type(Particles{Float64,10}, Missing) == Union{Particles{Float64,10},Missing}
        @testset "promotion of $PT" for PT in (Particles, StaticParticles)
            @test promote_type(PT{Float64,10}, PT{Float64,10}) == PT{Float64,10}
            @test promote_type(PT{Float64,10}, PT{Int,10}) == PT{Float64,10}
            @test promote_type(PT{Int,5}, PT{Float64,10}) == PT

            @test promote_type(PT{Float64,10}, PT{Float32,10}) == PT{Float64,10}
            @test promote_type(StaticParticles{Float64,10}, PT{Float32,10}) == StaticParticles{Float64,10}
        end
        @test promote_type(Particles{Float64,10}, StaticParticles{Float64,10}) == StaticParticles{Float64,10}
        @test promote_type(Particles{Int,10}, StaticParticles{Float64,10}) == StaticParticles{Float64,10}
        @test promote_type(Particles{Float64,10}, StaticParticles{Float32,10}) == StaticParticles{Float64,10}
        @test promote_type(Particles{Int,10}, StaticParticles{Float32,10}) == StaticParticles{Float32,10}
        @test convert(Float64, 0p) isa Float64
        @test convert(Float64, 0p) == 0
        @test convert(Int, 0p) isa Int
        @test convert(Int, 0p) == 0
        @test convert(Particles{Float64,100}, Particles(randn(Float32, 100))) isa Particles{Float64,100}
        @test_throws ArgumentError convert(Int, p)
        @test_throws ArgumentError AbstractFloat(p)
        @test AbstractFloat(0p) == 0.0
        @test Particles(500) + Particles(randn(Float32, 500)) isa typeof(Particles(500))
        @test isfinite(p)
        @test iszero(0p)
        @test iszero(p, 0.1)
        @test !iszero(p)
        @test !(!p)
        @test !(0p)
        @test round(p) ≈ 0 atol=0.1
        @test round(Int,p) == 0
        @test round(Int,p) isa Int
        @test sincos(p) == (sin(p), cos(p))
        @test norm(p) == abs(p)
        @test mean(norm([p,p]) - sqrt(2p^2)) < sqrt(eps()) # ≈ with atol fails on mac
        @test mean(LinearAlgebra.norm2([p,p]) - sqrt(2p^2)) < sqrt(eps()) # ≈ with atol fails on mac
        @test MvNormal(Particles(500, MvNormal(2,1))) isa MvNormal
        @test eps(typeof(p)) == eps(Float64)
        @test eps(p) == eps(Float64)
        A = randn(2,2)
        B = A .± 0
        @test sum(mean, exp(A) .- exp(B)) < 1e-9
        @test sum(mean, abs.(log(A)) .- abs.(log(B))) < 1e-9
        @test sum(mean, abs.(eigvals(A)) .- abs.(eigvals(B))) < 1e-9

        @test @unsafe mean(sum(abs, sort(eigvals([0 1 ± 0.001; -1. 0]), by=imag) - [ 0.0 - (1.0 ± 0.0005)*im
                                                0.0 + (1.0 ± 0.0005)*im])) < 0.002

        e = eigvals([1 ± 0.001 0; 0 1.])
        @test e isa Vector{Particles{Float64,DEFAULT_NUM_PARTICLES}}
        @test e ≈ [1.0 ± 0.00058, 1.0 ± 0.00058]

        @test (1 .. 2) isa Particles
        @test std(diff(sort((1 .. 2).particles))) < sqrt(eps())
        @test maximum((1 .. 2)) <= 2
        @test minimum((1 .. 2)) >= 1


        pp = [1. 0; 0 1] .± 0.0
        @test lyap(pp,pp) == lyap([1. 0; 0 1],[1. 0; 0 1])

        @test intersect(p,p) == union(p,p)
        @test length(intersect(p, 1+p)) < 2length(p)
        @test length(union(p, 1+p)) == 2length(p)

        p = 2 ± 0
        q = 3 ± 0
        @test sqrt(complex(p,p)) == sqrt(complex(2,2))
        @test exp(complex(p,p)) == exp(complex(2,2))
        @test sqrt!(fill(complex(1.,1.), DEFAULT_NUM_PARTICLES), complex(p,p)) == sqrt(complex(2,2))
        @test exp!(fill(complex(1.,1.), DEFAULT_NUM_PARTICLES), complex(p,p)) == exp(complex(2,2))
        y = Particles(100)
        @test exp(im*y) ≈ cos(y) + im*sin(y)
        @test complex(p,p)/complex(q,q) == complex(2,2)/complex(3,3)

        z = complex(1 ± 0.1, 1 ± 0.1)
        @unsafe @test abs(sqrt(z ^ 2) - z) < eps()
        @unsafe @test abs(sqrt(z ^ 2.0) - z) < eps()

        z = complex(1 ± 0.1, 0 ± 0.1)
        @test real(2 ^ z) ≈ 2 ^ real(z)
        @test real(2.0 ^ z) ≈ 2.0 ^ real(z)
        @test real(z ^ z) ≈ real(z) ^ real(z)
        p = 2 ± 0.1
        q = 3 ± 0.1
        @test wasserstein(p,p,1) == 0
        @test wasserstein(p,q,1) >= 0
        @test bootstrap(p) ≈ p
        rng = MersenneTwister(453)
        p1 = bootstrap(rng,p)
        rng = MersenneTwister(453)
        p2 = bootstrap(rng,p)
        @test p1 == p2
    end

    @time @testset "mutation" begin
        @info "Testing mutation"
        function adder!(x)
            for i = eachindex(x)
                x[i] += 1
            end
            x
        end
        x = (1:5) .± 1
        adder!(x)
        @test x ≈ ((2:6) .± 1)
    end

    @time @testset "outer_product" begin
        @info "Testing outer product"
        d = 2
        μ = zeros(d)
        σ = ones(d)
        p = μ ⊗ σ
        @test length(p) == 2
        @test length(p[1]) <= 100_000
        @test cov(p) ≈ I atol=1e-1
        p = μ ⊗ 1
        @test length(p) == 2
        @test length(p[1]) <= 100_000
        @test cov(p) ≈ I atol=1e-1
        p = 0 ⊗ σ
        @test length(p) == 2
        @test length(p[1]) <= 100_000
        @test cov(p) ≈ I atol=1e-1

        rng = MersenneTwister(38)
        p1 = outer_product(rng, Normal.(μ,σ))
        rng = MersenneTwister(38)
        p2 = outer_product(rng, Normal.(μ,σ))
        @test p1 == p2
    end

    @time @testset "plotting" begin
        @info "Testing plotting"
        p = 0 ± 1
        v = [p,p]
        @test_nowarn Plots.plot(p)
        @test_nowarn Plots.plot(v)
        @test_nowarn Plots.plot(x->x^2,v)
        @test_nowarn Plots.plot(v,v)
        @test_nowarn Plots.plot(v,v; points=true)
        @test_nowarn Plots.plot(v,ones(2))
        @test_nowarn Plots.plot(1:2,v)

        @test_nowarn errorbarplot(1:2,v)
        @test_nowarn mcplot(1:2,v)
        @test_nowarn ribbonplot(1:2,v)
        @test_nowarn ribbonplot(1:2,v,(0.1,0.9))

        @test_nowarn MonteCarloMeasurements.print_functions_to_extend()
    end

    @time @testset "optimize" begin
        @info "Testing optimization"
        function rosenbrock2d(x)
            return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        end
        @test any(1:10) do i
            p = -1ones(2) .+ 2 .*Particles.(200) # Optimum is in [1,1]
            popt = optimize(rosenbrock2d, deepcopy(p))
            popt ≈ [1,1]
        end

        p = -1ones(2) .+ 2 .*Particles.(200) # Optimum is in [1,1]
        rng = MersenneTwister(876)
        popt1 = optimize(rng, rosenbrock2d, deepcopy(p))
        rng = MersenneTwister(876)
        popt2 = optimize(rng, rosenbrock2d, deepcopy(p))
        @test popt1 == popt2
    end


    @time @testset "bymap" begin
        @info "Testing bymap"

        p = 0 ± 1
        ps = 0 ∓ 1

        f(x) = 2x
        f(x,y) = 2x + y

        @test f(p) ≈ bymap(f,p)
        @test f(p) ≈ @bymap f(p)
        @test typeof(f(p)) == typeof(@bymap(f(p)))
        @test typeof(f(ps)) == typeof(@bymap(f(ps)))
        @test f(p,p) ≈ @bymap f(p,p)
        @test f(p,10) ≈ @bymap f(p,10)
        @test !(f(p,10) ≈ @bymap f(p,-10))

        @test f(p) ≈ @bypmap f(p)
        @test f(p,p) ≈ @bypmap f(p,p)
        @test f(p,10) ≈ @bypmap f(p,10)
        @test !(f(p,10) ≈ @bypmap f(p,-10))

        g(x,y) = sum(x) + sum(y)
        @test g([p,p], [p,p]) ≈ @bymap g([p,p], [p,p])
        @test g([p,p], p) ≈ @bymap g([p,p], p)
        @test g([p p], [p,p]) ≈ @bymap g([p p], [p,p])

        @test g([p,p], [p,p]) ≈ @bypmap g([p,p], [p,p])
        @test g([p,p], p) ≈ @bypmap g([p,p], p)
        @test g([p p], [p,p]) ≈ @bypmap g([p p], [p,p])

        h(x,y) = x .* y'
        Base.Cartesian.@nextract 4 p d-> 0±1
        @test all(h([p_1,p_2], [p_3,p_4]) .≈ bymap(h, [p_1,p_2], [p_3,p_4]))
        @test all(h([p_1,p_2], [p_3,p_4]) .≈ bypmap(h, [p_1,p_2], [p_3,p_4]))

        h2(x,y) = x .* y
        Base.Cartesian.@nextract 4 p d-> 0±1
        @test h2([p_1,p_2], [p_3,p_4]) ≈ @bymap  h2([p_1,p_2], [p_3,p_4])
        @test h2([p_1,p_2], [p_3,p_4]) ≈ @bypmap h2([p_1,p_2], [p_3,p_4])

        g(nt::NamedTuple) = nt.x^2 + nt.y^2
        @test g((x=p_1, y=p_2)) == p_1^2 + p_2^2

        g2(a,nt::NamedTuple) = a + nt.x^2 + nt.y^2
        @test g2(p_3, (x=p_1, y=p_2)) == p_3 + p_1^2 + p_2^2

    end

    @testset "inference" begin
        @inferred zero(Particles{Float64,1})
        @inferred zeros(Particles{Float64,1}, 5)
        @inferred bymap(sin, 1 ± 2)
    end

    include("test_forwarddiff.jl")
    include("test_deconstruct.jl")
    include("test_sleefpirates.jl")
    include("test_measurements.jl")

end

# These can not be inside a testset, causes "testf not defined"
testf(x,y) = sum(x+y)
@test_nowarn register_primitive(testf)
p = 1 ± 0.1
@test testf(p,p) == sum(p+p)



# Integration tests and bechmarks

# using BenchmarkTools
# A = [StaticParticles(100) for i = 1:3, j = 1:3]
# B = similar(A, Float64)
# @btime qr($(copy(A)))
# @btime map(_->qr($B), 1:100);

#
# # Benchmark and comparison to Measurements.jl
# using BenchmarkTools, Printf, ControlSystems
# using MonteCarloMeasurements, Measurements
# using Measurements: ±
# using MonteCarloMeasurements: ∓
# w = exp10.(LinRange(-0.7,0.3,50))
#
# p = 1 ± 0.1
# ζ = 0.3 ± 0.1
# ω = 1 ± 0.1
# Gm = tf([p*ω], [1, 2ζ*ω, ω^2])
# # tm = @belapsed bode($Gm,$w)
#
# p = 1 ∓ 0.1
# ζ = 0.3 ∓ 0.1
# ω = 1 ∓ 0.1
# Gmm = tf([p*ω], [1, 2ζ*ω, ω^2])
# # tmm = @belapsed bode($Gmm,$w)
#
# σquant = 1-(cdf(Normal(0,1), 1)-cdf(Normal(0,1), -1))
#
# magm = bode(Gm,w)[1][:]
# magmm = bode(Gmm,w)[1][:]
# errorbarplot(w,magmm, σquant/2, xscale=:log10, yscale=:log10, lab="Particles", linewidth=2)
# plot!(w,magm, lab="Measurements")
