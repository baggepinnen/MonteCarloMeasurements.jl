@info "Running tests"
using MonteCarloMeasurements, Distributions
using Test, LinearAlgebra, Statistics, Random, GenericSchur
import MonteCarloMeasurements: ⊗, gradient, optimize, DEFAULT_NUM_PARTICLES, vecindex
@info "import Plots, Makie"
import Plots
import Makie
@info "import plotting packages done"

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
        params = Distributions.params
        @test systematic_sample(10000, Normal(1,1)) |> Base.Fix1(fit, Normal) |> params |> x-> all(isapprox.(x,(1,1), atol=0.1))
        systematic_sample(10000, Gamma(1,1)) #|> Base.Fix1(fit, Gamma)
        systematic_sample(10000, TDist(1)) #|> Base.Fix1(fit, TDist)
        @test systematic_sample(10000, Beta(1,1)) |> Base.Fix1(fit, Beta) |> params |> x-> all(isapprox.(x,(1,1), atol=0.1))

        for i = 1:100
            @test MonteCarloMeasurements.ess(Particles(10000)) > 7000
            x = randn(5000)
            v = vec([x';x'])
            @test 3000 < MonteCarloMeasurements.ess(Particles(v)) < 5300
        end
    end
    
    @time @testset "Basic operations" begin
        @info "Creating the first StaticParticles"
        @test 0 ∓ 1 isa StaticParticles
        @test [0,0] ∓ 1. isa MonteCarloMeasurements.MvParticles
        @test [0,0] ∓ [1.,1.] isa MonteCarloMeasurements.MvParticles
        @test -50 ⊞ Normal(0,1) ≈ -50 ± 1
        @test 10 ⊠ Normal(0,1) ≈ 10*Particles(Normal(0,1))

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
                @test ± 1 ≈ p
                @test 0 ∓ 1 ≈ p
                @test ∓ 1 ≈ p
                @test sum(p) ≈ 0
                @test pcov(p) ≈ 1 atol=0.2
                @test pstd(p) ≈ 1 atol=0.2
                @test pvar(p) ≈ 1 atol=0.2
                @test meanvar(p) ≈ 1/(nparticles(p)) atol=5e-3
                @test meanstd(p) ≈ 1/sqrt(nparticles(p)) atol=5e-3
                @test minmax(1+p,p) == (p, 1+p)
                b = PT(100)
                uvec = unique([p, p, b, p, b, b]) # tests hash
                @test length(uvec) == 2
                @test p ∈ uvec
                @test b ∈ uvec

                @test PT(100, Normal(0.0)) isa PT{Float64, 100}
                @test PT(100, Normal(0.0f0)) isa PT{Float32, 100}
                @test PT(100, Uniform(0.0f0, 1.0f0)) isa PT{Float32, 100}


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
                @test p ≈ 1.9pstd(p)
                @test !(p ≈ 2.1pstd(p))
                @test !(p ≉ p)
                @test !(pmean(p) ≉ p)
                @test p ≉ 2.1pstd(p)
                @test !(p ≉ 1.9pstd(p))
                @test (5 ± 0.1) ≳ (1 ± 0.1)
                @test (1 ± 0.1) ≲ (5 ± 0.1)

                a = rand()
                pa = Particles([a])
                @test a == pa
                @test a ≈ pa
                @test pa ≈ a

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
                    @test_throws ErrorException p < Particles(p.particles[randperm(nparticles(p))])
                    unsafe_comparisons(false)

                    @unsafe tv = 2
                    @test tv == 2
                    @unsafe tv1,tv2 = 1,2
                    @test (tv1,tv2) == (1,2)
                    @unsafe tv3,tv4 = range(1, stop=3, length=5), range(1, stop=3, length=5)
                    @test (tv3,tv4) == (range(1, stop=3, length=5),range(1, stop=3, length=5))
                    @test MonteCarloMeasurements.COMPARISON_FUNCTION[] == pmean
                    set_comparison_function(pmedian)
                    @test MonteCarloMeasurements.COMPARISON_FUNCTION[] == pmedian
                    cp = PT(10)
                    cmp = @unsafe p < cp
                    @test cmp == (pmedian(p) < pmedian(cp))
                    set_comparison_function(pmean)
                end


                f = x -> 2x + 10
                @test 9.6 < pmean(f(p)) < 10.4
                # @test 9.6 < f(p) < 10.4
                @test f(p) ≈ 10
                @test !(f(p) ≲ 11)
                @test f(p) ≲ 15
                @test 5 ≲ f(p)
                @test 1 ≲ 2

                @test Normal(f(p)).μ ≈ pmean(f(p))
                @test fit(Normal, f(p)).μ ≈ pmean(f(p))


                f = x -> x^2
                f3 = x -> x^3
                p = PT(100)
                @test 0.9 < pmean(f(p)) < 1.1
                @test 0.9 < pmean(f(p)) < 1.1
                @test f(p) ≈ 1
                @test f3(p) ≈ 1
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
                @test [1,1,1] ≈ A*b .+ 1

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

                @test PT(Float32) isa PT{Float32}
                @test PT(Float64) isa PT{Float64}
                @test PT(Float32, 250) isa PT{Float32, 250}
                @test PT(Float32, 250, Normal(0.1f0)) isa PT{Float32, 250}
                @test_throws ArgumentError PT(Float32, 250, Gamma(0.1))

                @info "Tests for $PT done"

                p = PT{Float64,10}(2)
                @test p isa PT{Float64,10}
                @test all(p.particles .== 2)

                @test Particles(100) + Particles(randn(Float32, 100)) ≈ 0
                @test_throws MethodError p + Particles(randn(Float32, 200)) # Npart and Float type differ
                @test_throws MethodError p + Particles(200) # Npart differ

                mp = MvParticles([randn(10) for _ in 1:3])
                @test length(mp) == 10
                @test nparticles(mp) == 3

                v = [(1,2), (3,4)]
                pv = MvParticles(v)
                @test length(pv) == 2
                @test pv[1].particles[1] == v[1][1]
                @test pv[1].particles[2] == v[2][1]
                @test pv[2].particles[1] == v[1][2]
                @test pv[2].particles[2] == v[2][2]
                @test length(pv) == 2

                v = [(a=1,b=randn(2)), (a=3,b=randn(2))]
                pv = MvParticles(v)
                @test length(pv) == 2
                @test pv.a.particles[1] == v[1].a
                @test pv.a.particles[2] == v[2].a
                @test pv.b == Particles([v[1].b v[2].b]')
                @test length(pv) == 2

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
                @test pcov(p) ≈ I atol=0.6
                @test pcor(p) ≈ I atol=0.6
                @test pmean(p) ≈ [0,0] atol=0.2
                m = Matrix(p)
                @test size(m) == (100,2)
                # @test m[1,2] == p[1,2]

                p = PT(100, MvNormal(2,2))
                @test pcov(p) ≈ 4I atol=2
                @test [0,0] ≈ pmean(p) atol=1
                @test size(Matrix(p)) == (100,2)

                p = PT(100, MvNormal(2,2))
                @test fit(MvNormal, p).μ ≈ pmean(p)
                @test MvNormal(p).μ ≈ pmean(p)
                @test cov(MvNormal(p)) ≈ pcov(p)
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
        @test sum(abs, tr((pcov(xhp) .- C1) ./ abs.(C1))) < 0.2

        @test norm(pcov(xhp) .- C1) < 1e-7
        @test xhp ≈ x
        @test pmean(xhp) ≈ x atol=3sum(sqrt.(diag(C1)))
        yp = nothing; GC.gc(true) # make sure this big matrix is deallocated
    end

    @time @testset "misc" begin
        @info "Testing misc"
        p = 0 ± 1
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
        @test_nowarn show(stdout, MIME"text/plain"(), -im*p -10im); println()

        @test_nowarn display([p, p])
        @test_nowarn println([p, p])
        @test_nowarn println([p, 0p])

        @test Particles{Float64,DEFAULT_NUM_PARTICLES}(p) == p
        @test Particles{Float64,5}(0) == 0*Particles(5)
        @test length(Particles(100, MvNormal(2,1))) == 2
        @test nparticles(p) == DEFAULT_NUM_PARTICLES
        @test ndims(p) == 0
        @test particleeltype(p) == Float64
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
        @test round(p) == Particles(round.(p.particles)) 
        @test round(Int,0.001p) == 0
        @test round(Int,p) isa Particles{Int, nparticles(p)}
        @test sincos(p) == (sin(p), cos(p))
        @test norm(p) == abs(p)
        @test pmean(norm([p,p]) - sqrt(2p^2)) < sqrt(eps()) # ≈ with atol fails on mac
        @test pmean(LinearAlgebra.norm2([p,p]) - sqrt(2p^2)) < sqrt(eps()) # ≈ with atol fails on mac
        @test MvNormal(Particles(500, MvNormal(2,1))) isa MvNormal
        @test eps(typeof(p)) == eps(Float64)
        @test eps(p) == eps(Float64)
        A = randn(2,2)
        B = A .± 0
        @test sum(pmean, exp(A) .- exp(B)) < 1e-9
        @test sum(pmean, LinearAlgebra.exp!(copy(A)) .- LinearAlgebra.exp!(copy(B))) < 1e-9
        @test sum(pmean, abs.(log(A)) .- abs.(log(B))) < 1e-9
        @test sum(pmean, abs.(eigvals(A)) .- abs.(eigvals(B))) < 1e-9

        @test @unsafe pmean(sum(abs, sort(eigvals([0 1 ± 0.001; -1. 0]), by=imag) - [ 0.0 - (1.0 ± 0.0005)*im
                                                0.0 + (1.0 ± 0.0005)*im])) < 0.002

        e = eigvals([1 ± 0.001 0; 0 1.])
        @test e isa Vector{Complex{Particles{Float64, DEFAULT_NUM_PARTICLES}}}
        @test all(isapprox.(e, [1.0 ± 0.00058, 1.0 ± 0.00058], atol=1e-2))

        ## Complex matrix ops
        A = randn(ComplexF64, 2, 2)
        B = complex.(Particles.(fill.(real.(A), 10)), Particles.(fill.(imag.(A), 10)))
        show(B)
        @test sum(pmean, abs.(exp(A) .- exp(B))) < 1e-9
        @test sum(pmean, abs.(LinearAlgebra.exp!(copy(A)) .- LinearAlgebra.exp!(copy(B)))) < 1e-9
        @test sum(pmean, abs.(log(A) .- log(B))) < 1e-9
        @test abs(det(A) - det(B)) < 1e-9

        @test sum(pmean, svdvals(A) .- svdvals(B)) < 1e-9
        @test sum(pmean, abs.(eigvals(A) .- eigvals(B))) < 1e-9

        @test (1 .. 2) isa Particles
        @test std(diff(sort((1 .. 2).particles))) < sqrt(eps())
        @test pmaximum((1 .. 2)) <= 2
        @test pminimum((1 .. 2)) >= 1


        pp = [1. 0; 0 1] .± 0.0
        @test lyap(pp,pp) == lyap([1. 0; 0 1],[1. 0; 0 1])

        @test intersect(p,p) == union(p,p)
        @test nparticles(intersect(p, 1+p)) < 2nparticles(p)
        @test nparticles(union(p, 1+p)) == 2nparticles(p)

        p = 2 ± 0
        q = 3 ± 0
        @test sqrt(complex(p,p)) == sqrt(complex(2,2))
        @test exp(complex(p,p)) == exp(complex(2,2))
        @test sqrt!(fill(complex(1.,1.), DEFAULT_NUM_PARTICLES), complex(p,p)) == sqrt(complex(2,2))
        @test exp!(fill(complex(1.,1.), DEFAULT_NUM_PARTICLES), complex(p,p)) == exp(complex(2,2))
        y = Particles(100)
        @test exp(im*y) ≈ cos(y) + im*sin(y)
        @test complex(p,p)/complex(q,q) == complex(2,2)/complex(3,3)
        @test p/complex(q,q) == 2/complex(3,3)
        @test Base.FastMath.div_fast(p, complex(q,q)) == Base.FastMath.div_fast(2, complex(3,3))
        @test 2/complex(q,q) == 2/complex(3,3)
        @test !isinf(complex(p,p))
        @test isfinite(complex(p,p))

        z = complex(1 ± 0.1, 1 ± 0.1)
        @unsafe @test abs(sqrt(z ^ 2) - z) < eps()
        @unsafe @test abs(sqrt(z ^ 2.0) - z) < eps()
        @test z/z ≈ 1

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
        @test nparticles(bootstrap(p, 10)) == 10
        @test_nowarn bootstrap([p; p])

        dict = Dict(:a => p, :b => q, :c => 1)
        dictvec = MonteCarloMeasurements.particle_dict2dict_vec(dict)
        @test length(dictvec) == nparticles(p)
        @test dictvec[1] == Dict(:a => p.particles[1], :b => q.particles[1], :c => 1)
        @test dictvec[2] == Dict(:a => p.particles[2], :b => q.particles[2], :c => 1)

        H = (10± 0.0) .*rand(ComplexF64,3,2)
        R1 = qr(mean_object(H)).R
        R2 = MonteCarloMeasurements.ℂⁿ2ℂⁿ_function(x->(qr(x).R),H)
        @test R1 ≈ mean_object(R2)

    end

    @testset "vecindex tests" begin
        # Test for Complex{<:AbstractParticles}
        p_real = Particles(100, Normal(0, 1))
        p_imag = Particles(100, Normal(0, 1))
        p_complex = complex(p_real, p_imag)
        
        @test vecindex(p_complex, 1) == complex(p_real.particles[1], p_imag.particles[1])
        @test vecindex(p_complex, 50) == complex(p_real.particles[50], p_imag.particles[50])
        @test vecindex(p_complex, 100) == complex(p_real.particles[100], p_imag.particles[100])
    
        # Test for AbstractArray{<:Complex{<:AbstractParticles}}
        p_array = [p_complex, p_complex]
        
        @test vecindex(p_array, 1) == [complex(p_real.particles[1], p_imag.particles[1]), complex(p_real.particles[1], p_imag.particles[1])]
        @test vecindex(p_array, 50) == [complex(p_real.particles[50], p_imag.particles[50]), complex(p_real.particles[50], p_imag.particles[50])]
        @test vecindex(p_array, 100) == [complex(p_real.particles[100], p_imag.particles[100]), complex(p_real.particles[100], p_imag.particles[100])]
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
        @test nparticles(p[1]) <= 100_000
        @test pcov(p) ≈ I atol=1e-1
        p = μ ⊗ 1
        @test length(p) == 2
        @test nparticles(p[1]) <= 100_000
        @test pcov(p) ≈ I atol=1e-1
        p = 0 ⊗ σ
        @test length(p) == 2
        @test nparticles(p[1]) <= 100_000
        @test pcov(p) ≈ I atol=1e-1

        rng = MersenneTwister(38)
        p1 = outer_product(rng, Normal.(μ,σ))
        rng = MersenneTwister(38)
        p2 = outer_product(rng, Normal.(μ,σ))
        @test p1 == p2
    end

    @time @testset "plotting" begin
        @info "Testing plotting"
        p = Particles(100)
        v = randn(3) .+ Particles.(10)
        M = randn(3,2) .+ [-1 1] .+ Particles.(10)
        @test_nowarn Plots.plot(p)
        @test_nowarn Plots.plot(v)
        @test_nowarn Plots.plot(M)
        @test_nowarn Plots.plot(M .+ 5) # This plot should have 4 different colored bands
        @test_nowarn Plots.plot(v, ri=false)
        @test_nowarn Plots.plot(v, N=0)
        @test_nowarn Plots.plot(M, N=0)
        @test_nowarn Plots.plot(v, N=8)
        @test_nowarn Plots.plot(M, N=10)
        @test_nowarn Plots.plot(x->x^2,v)
        @test_nowarn Plots.plot(v,v)
        @test_nowarn Plots.plot(v,v, N=10)
        @test_nowarn Plots.plot(v,v; points=true)
        @test_nowarn Plots.plot(v,ones(3))
        @test_nowarn Plots.plot(1:3,v)
        @test_nowarn Plots.plot(1:3,v, ri=false)
        @test_nowarn Plots.plot(1:3, v, N=5)
        @test_nowarn Plots.plot(1:3, M, N=5)
        @test_nowarn Plots.plot!(1:3, M .+ 5, N=5) # This plot should have 4 different colored bands
        @test_nowarn Plots.plot((1:3) .* [1 1], M, N=10)

        @test_nowarn Plots.plot(1:3, M, N=5, ri=false)
        @test_nowarn Plots.plot!(1:3, M .+ 5, N=5, ri=false) # This plot should have 4 different colored mclines

        @test_nowarn Plots.plot(1:3, v, N=0)
        @test_nowarn Plots.plot(1:3, M, N=0)
        @test_nowarn Plots.plot!(1:3, M .+ 5, N=0) # This plot should have 4 different colored bands
        @test_nowarn Plots.plot((1:3) .* [1 1], M, N=0)

        @test_nowarn errorbarplot(1:3,v)
        @test_nowarn errorbarplot(1:3,[v v])
        @test_nowarn mcplot(1:3,v)
        @test_nowarn mcplot(1:3,v, 10)
        @test_nowarn mcplot(1:3,[v v])
        @test_nowarn ribbonplot(1:3,v)
        @test_nowarn ribbonplot(1:3,v,(0.1,0.9))

        @test_nowarn errorbarplot(v)
        @test_nowarn errorbarplot([v v])
        @test_nowarn mcplot(v)
        @test_nowarn mcplot(v, 10)
        @test_nowarn mcplot([v v])
        @test_nowarn ribbonplot(v)
        @test_nowarn ribbonplot(v,(0.1,0.9))
        @test_nowarn ribbonplot(v, N = 2)

        @test_nowarn errorbarplot(v, 0.1)
        @test_nowarn errorbarplot([v v], 0.1)
        @test_nowarn mcplot(v, 0.1)
        @test_nowarn ribbonplot(v, 0.1)

        @test_throws ArgumentError errorbarplot(1:3, (1:3) .± 0.1, 1,1)

        @test_nowarn MonteCarloMeasurements.print_functions_to_extend()
    end

    @time @testset "Makie" begin
        p1 = Particles(10^2)
        Makie.hist(p1)
        Makie.density(p1)

        xs = 1:20
        ys = Particles.(Normal.(sqrt.(1:20), sqrt.(1:20)./5))

        Makie.scatter(xs, ys)
        Makie.scatter(tuple.(xs, ys))
        Makie.band(xs, ys)
        Makie.band(tuple.(xs, ys); q=0.01)
        Makie.band(tuple.(xs, ys); nσ=2)
        Makie.rangebars(tuple.(xs, ys); q=0.16)
        @test_throws "Only one of" Makie.rangebars(tuple.(xs, ys); q=0.16, nσ=2)
        Makie.series(xs, ys)
        Makie.series(tuple.(xs, ys); N=5)
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


    @testset "Particle BLAS" begin
        @info "Testing Particle BLAS"
        p = ones(10) .∓ 1
        A = randn(20,10)
        @test pmean(sum(abs, A*p - MonteCarloMeasurements._pgemv(A,p))) < 1e-12
        @test pmean(sum(abs, A*Particles.(p) - A*p)) < 1e-12

        v = randn(10)
        @test pmean(sum(abs, v'p - MonteCarloMeasurements._pdot(v,p))) < 1e-12
        @test pmean(sum(abs, p'v - MonteCarloMeasurements._pdot(v,p))) < 1e-12
        @test pmean(sum(abs, v'*Particles.(p) - v'p)) < 1e-12


        @test pmean(sum(abs, axpy!(2.0,Matrix(p),copy(Matrix(p))) - Matrix(copy(axpy!(2.0,p,copy(p)))))) < 1e-12
        @test pmean(sum(abs, axpy!(2.0,Matrix(p),copy(Matrix(p))) - Matrix(copy(axpy!(2.0,p,copy(p)))))) < 1e-12


        y = randn(20) .∓ 1
        @test pmean(sum(abs, mul!(y,A,p) - mul!(Particles.(y),A,Particles.(p)))) < 1e-12

        for PT in (Particles, StaticParticles)
            for x in (1.0, 1 + PT()), y in (1.0, 1 + PT()), z in (1.0, 1 + PT())
                x == y == z == 1.0 && continue
                @test (x*y+z).particles ≈ muladd(x,y,z).particles
            end
        end

        x = 1.0 + Particles()
        y = 1.0 + 2im
        z = 1.0 + Particles()
        @test (x*y+z).re.particles ≈ muladd(x,y,z).re.particles
        @test (x*y+z).im.particles ≈ muladd(x,y,z).im.particles


        #
        # @btime $A*$p
        # @btime _pgemv($A,$p)
        #
        # @btime sum($A*$p)
        # @btime sum(_pgemv($A,$p))
        #
        # @btime $v'*$p
        # @btime _pdot($v,$p)
        #
        # @btime sum($v'*$p)
        # @btime sum(_pdot($v,$p))

        # @btime mul!($y,$A,$p)
        # @btime MonteCarloMeasurements.pmul!($y,$A,$p)
        # 178.373 μs (6 allocations: 336 bytes)
        # 22.320 μs (0 allocations: 0 bytes)
        # 3.705 μs (0 allocations: 0 bytes)
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

        @test_throws ErrorException bymap(x->ones(3,3,3,3), p)
        @test_throws ErrorException bypmap(x->ones(3,3,3,3), p)
        @test MonteCarloMeasurements.arggetter(1,1) == 1

        @test MonteCarloMeasurements.particletype(p) == Particles{Float64,DEFAULT_NUM_PARTICLES}
        @test MonteCarloMeasurements.particletype(Particles{Float64,DEFAULT_NUM_PARTICLES}) == Particles{Float64,DEFAULT_NUM_PARTICLES}
        @test MonteCarloMeasurements.particletype([p,p]) == Particles{Float64,DEFAULT_NUM_PARTICLES}
        @test nparticles(p) == DEFAULT_NUM_PARTICLES




        @testset "@prob" begin
            @info "Testing @prob"
            p = Particles()
            q = Particles()

            @test mean((p).particles .< 1) == @prob p < 1
            @test mean((p+2).particles .< 1) == @prob p+2 < 1
            @test mean((2+p).particles .< 1) == @prob 2+p < 1
            @test mean((p+p).particles .< 1) == @prob p+p < 1
            @test mean((p).particles .< p.particles) == @prob p < p
            @test mean((p+1).particles .< p.particles) == @prob p+1 < p

            @test mean((p+q).particles .< 1) == @prob p+q < 1
            @test mean((p).particles .< q.particles) == @prob p < q
            @test mean((p+1).particles .< q.particles) == @prob p+1 < q

            @test mean((p+q).particles .> 1) == @prob p+q > 1
            @test mean((p).particles .> q.particles) == @prob p > q
            @test mean((p+1).particles .> q.particles) == @prob p+1 > q

            @test mean((p+q).particles .>= 1) == @prob p+q >= 1
            @test mean((p).particles .>= q.particles) == @prob p >= q
            @test mean((p+1).particles .>= q.particles) == @prob p+1 >= q

            @test mean((abs(p)).particles .> sin(q).particles) == @prob abs(p) > sin(q)

        end

    end

    @testset "inference" begin
        @inferred zero(Particles{Float64,1})
        @inferred zeros(Particles{Float64,1}, 5)
        @inferred bymap(sin, 1 ± 2)
    end

    @testset "nominal values" begin
        @info "Testing nominal values"

        p = 1 ± 0.1
        n = 0
        pn = with_nominal(p, n)
        @test nominal(pn) == pn.particles[1] == n
        @test nominal(p) != n

        p = [p, p]
        n = [0, 1]
        pn = with_nominal(p, n)
        @test nominal(pn) == MonteCarloMeasurements.vecindex.(pn, 1) == n
        @test nominal(p) != n


        p = 1 ∓ 0.1
        n = 0
        pn = with_nominal(p, n)
        @test nominal(pn) == pn.particles[1] == n
        @test nominal(p) != n

        p = [p, p]
        n = [0, 1]
        pn = with_nominal(p, n)
        @test nominal(pn) == MonteCarloMeasurements.vecindex.(pn, 1) == n
        @test nominal(p) != n

        P = complex(1 ± 0.1, 2 ± 0.1)
        @test nominal(P) == complex(real(P).particles[1], imag(P).particles[1])

    end

    include("test_unitful.jl")
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
