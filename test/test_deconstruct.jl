
using MonteCarloMeasurements: nakedtypeof, build_container, build_mutable_container, has_particles, particle_paths, get_setter_funs
using ControlSystems, Test
ControlSystems.TransferFunction(matrix::Array{<:ControlSystems.SisoRational,2}, Ts::Float64, ::Int64, ::Int64) = TransferFunction(matrix,Ts)


@testset "deconstruct" begin
    unsafe_comparisons()
    P = tf(1 ∓ 0.1, [1, 1∓0.1])
    w = Workspace(P)
    f = x->c2d(x,0.1)
    @time Pd = w(f)

    tt = function (P)
        w = Workspace(P)
        f = x->c2d(x,0.1)
        @time Pd = w(f)
    end
    @test_throws MethodError tt(P) # This causes a world-age problem. If this tests suddenly break, it would be nice and we can get rid of the intermediate workspace object.

    @test nakedtypeof(P) == TransferFunction
    @test nakedtypeof(typeof(P)) == TransferFunction
    @test typeof(P) == TransferFunction{ControlSystems.SisoRational{StaticParticles{Float64,100}}}
    P2 = build_container(P)
    @test typeof(P2) == TransferFunction{ControlSystems.SisoRational{Float64}}
    @test typeof(build_mutable_container(P)) == TransferFunction{ControlSystems.SisoRational{Particles{Float64,100}}}
    @test has_particles(P)
    @test has_particles(P.matrix)
    @test has_particles(P.matrix[1])
    @test has_particles(P.matrix[1].num)
    @test has_particles(P.matrix[1].num.a)
    @test has_particles(P.matrix[1].num.a[1])


    # P = tf(1 ± 0.1, [1, 1±0.1])
    # @benchmark foreach(i->c2d($(tf(1.,[1., 1])),0.1), 1:100)

    bP  = bode(P, exp10.(LinRange(-3, log10(10π), 50)))[1] |> vec
    bPd = bode(Pd, exp10.(LinRange(-3, log10(10π), 50)))[1] |> vec

    @test mean(abs2, mean.(bP) - mean.(bPd)) < 1e-4

    # bodeplot(P, exp10.(LinRange(-3, log10(10π), 50)))
    # bodeplot!(Pd, exp10.(LinRange(-3, log10(10π), 50)), linecolor=:blue)


    let paths = particle_paths(P), P2 = build_container(P), (setters, setters2) = get_setter_funs(paths)
        Pres = @unsafe build_mutable_container(f(P)) # Auto-created result buffer
        @test all(1:paths[1][3]) do i
            setters(P,P2,i)
            P.matrix[1].num.a[1][i] == P2.matrix[1].num.a[1] &&
            P.matrix[1].den.a[2][i] == P2.matrix[1].den.a[2]
            P2res = f(P2)
            setters2(Pres, P2res, i)
            Pres.matrix[1].num.a[1][i] == P2res.matrix[1].num.a[1] &&
            Pres.matrix[1].den.a[2][i] == P2res.matrix[1].den.a[2]
        end
    end
    unsafe_comparisons(false)
end
