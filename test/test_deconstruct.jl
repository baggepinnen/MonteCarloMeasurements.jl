using MonteCarloMeasurements
using Test, LinearAlgebra, Statistics, Random
import MonteCarloMeasurements: ±, ∓
using MonteCarloMeasurements: nakedtypeof, build_container, build_mutable_container, has_particles, particle_paths
using ControlSystems, Test
ControlSystems.TransferFunction(matrix::Array{<:ControlSystems.SisoRational,2}, Ts::Float64, ::Int64, ::Int64) = TransferFunction(matrix,Ts)


@testset "deconstruct" begin
    unsafe_comparisons()
    N = 50
    P = tf(1 +0.1StaticParticles(N), [1, 1+0.1StaticParticles(N)])
    f = x->c2d(x,0.1)
    w = Workspace(f,P)
    @time Pd = w(P)
    @test !MonteCarloMeasurements.has_mutable_particles(Pd)
    @test MonteCarloMeasurements.has_mutable_particles(MonteCarloMeasurements.build_mutable_container(Pd))
    # See benchmarks below
    # @profiler Workspace(P)

    tt = function (P)
        f = x->c2d(x,0.1)
        w = Workspace(f,P)
        @time Pd = w(P)
    end
    @test_throws MethodError tt(P) # This causes a world-age problem. If this tests suddenly break, it would be nice and we can get rid of the intermediate workspace object.
    tt = function (P)
        f = x->c2d(x,0.1)
        w = Workspace(f,P)
        @time Pd = w(P,true)
    end
    @test tt(P) == Pd == with_workspace(f,P)
    p = 1 ± 0.1
    @test mean_object(p) == mean(p)
    @test mean_object([p,p]) == mean.([p,p])
    @test mean_object(P) ≈ tf(tf(1,[1,1])) atol=1e-2



    @test nakedtypeof(P) == TransferFunction
    @test nakedtypeof(typeof(P)) == TransferFunction
    @test typeof(P) == TransferFunction{ControlSystems.SisoRational{StaticParticles{Float64,N}}}
    P2 = build_container(P)
    @test typeof(P2) == TransferFunction{ControlSystems.SisoRational{Float64}}
    @test typeof(build_mutable_container(P)) == TransferFunction{ControlSystems.SisoRational{Particles{Float64,N}}}
    @test has_particles(P)
    @test has_particles(P.matrix)
    @test has_particles(P.matrix[1])
    @test has_particles(P.matrix[1].num)
    @test has_particles(P.matrix[1].num.a)
    @test has_particles(P.matrix[1].num.a[1])

    # P = tf(1 ± 0.1, [1, 1±0.1])
    # @benchmark foreach(i->c2d($(tf(1.,[1., 1])),0.1), 1:N) # 1.7 ms 1.2 Mb

    bP  = bode(P, exp10.(LinRange(-3, log10(10π), 50)))[1] |> vec
    bPd = bode(Pd, exp10.(LinRange(-3, log10(10π), 50)))[1] |> vec

    @test mean(abs2, mean.(bP) - mean.(bPd)) < 1e-4

    A = randn(2,2)
    Ap = A .± 0.1
    hfun = A->Matrix(hessenberg(A))
    @test all(ℝⁿ2ℝⁿ_function(hfun, Ap) .≈ Matrix(hessenberg(A)))

    Ap = A .+ 0.1 .* StaticParticles(1)
    @test_nowarn hessenberg(Ap)

    # bodeplot(P, exp10.(LinRange(-3, log10(10π), 50)))
    # bodeplot!(Pd, exp10.(LinRange(-3, log10(10π), 50)), linecolor=:blue)


    let paths = particle_paths(P), P2 = build_container(P), buffersetter = MonteCarloMeasurements.get_buffer_setter(paths)
        Pres = @unsafe build_mutable_container(f(P)) # Auto-created result buffer
        resultsetter = MonteCarloMeasurements.get_result_setter(Pres)
        @test all(1:paths[1][3]) do i
            buffersetter(P,P2,i)
            P.matrix[1].num.a[1][i] == P2.matrix[1].num.a[1] &&
            P.matrix[1].den.a[2][i] == P2.matrix[1].den.a[2]
            P2res = f(P2)
            resultsetter(Pres, P2res, i)
            Pres.matrix[1].num.a[1][i] == P2res.matrix[1].num.a[1] &&
            Pres.matrix[1].den.a[2][i] == P2res.matrix[1].den.a[2]
        end
    end
    unsafe_comparisons(false)
end








# julia> @benchmark Pd = w(f) # different world age
# BenchmarkTools.Trial:
#   memory estimate:  1.63 MiB
#   allocs estimate:  19178
#   --------------
#   minimum time:     2.101 ms (0.00% GC)
#   median time:      2.199 ms (0.00% GC)
#   mean time:        2.530 ms (10.36% GC)
#   maximum time:     7.969 ms (53.42% GC)
#   --------------
#   samples:          1973
#   evals/sample:     1

# julia> @benchmark Pd = w(f)  # invokelatest
#   BenchmarkTools.Trial:
# memory estimate:  1.64 MiB
# allocs estimate:  19378
# --------------
# minimum time:     2.204 ms (0.00% GC)
# median time:      2.742 ms (0.00% GC)
# mean time:        3.491 ms (13.77% GC)
# maximum time:     17.103 ms (80.96% GC)
# --------------
# samples:          1429
# evals/sample:     1
#
# julia> @benchmark with_workspace($f,$P) # It seems the majority of the time is spent building the workspace object, so invokelatest really isn't that expensive.
# BenchmarkTools.Trial:
#   memory estimate:  7.90 MiB
#   allocs estimate:  148678
#   --------------
#   minimum time:     158.073 ms (0.00% GC)
#   median time:      165.134 ms (0.00% GC)
#   mean time:        165.842 ms (1.75% GC)
#   maximum time:     180.133 ms (4.80% GC)
#   --------------
#   samples:          31
#   evals/sample:     1
