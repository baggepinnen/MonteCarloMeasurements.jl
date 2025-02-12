using Unitful, Test, MonteCarloMeasurements
function unitful_testfunction(Vi)
    if Vi ≤ 0.0u"V"
        return 0.0u"V"
    elseif Vi ≥ 1.0u"V"
        return 1.0u"V"
    else
        return Vi
    end
end

register_primitive(unitful_testfunction) # must be outside testset

@testset "Unitful" begin
    @info "Testing Unitful"

    PT = Particles
    for PT in (Particles, StaticParticles)
        p1 = PT(100, Uniform(-0.5,1.5)) * 1u"V"
        p2 = PT(100, Uniform(-0.5,1.5)) * u"V"
        @test_nowarn println(p1)
        @test_nowarn display(p1)

        @test_nowarn println(0p1)
        @test_nowarn display(0p1)

        @test typeof(p1) == typeof(p2)

        p3 = unitful_testfunction(p1)
        @test pextrema(p3) == (0.0u"V", 1.0u"V")

        @test (1 ± 0.5)u"m" * (1 ± 0)u"kg" ≈ (1 ± 0.5)u"kg*m"
        @test (1 ± 0.5)u"m" * 1u"kg" ≈ (1 ± 0.5)u"kg*m"

        @test (1 ± 0.5)u"m" / (1 ± 0)u"kg" ≈ (1 ± 0.5)u"m/kg"
        @test (1 ± 0.5)u"m" / 1u"kg" ≈ (1 ± 0.5)u"m/kg"

        @test (1 ± 0.5)u"m" + (1 ± 0)u"m" ≈ (2 ± 0.5)u"m"
        @test (1 ± 0.5)u"m" + 1u"m" ≈ (2 ± 0.5)u"m"

        @test 1u"m" * (1 ± 0.5)u"kg" ≈ (1 ± 0.5)u"kg*m"
        @test 1u"m" / (1 ± 0.5)u"kg" ≈ (1 ± 0.5)u"m/kg"
        @test 1u"m" + (1 ± 0.5)u"m" ≈ (2 ± 0.5)u"m"

        typeof(promote(1u"V", (1.0 ± 0.1)u"V")) <: Tuple{Particles{<:Quantity}, Particles{<:Quantity}}

        @test muladd(p1, 1, p1) == p1 + p1
        @test muladd(p1, 1, p2) == p1 + p2
        @test muladd(1, p1, p2) == p1 + p2
        @test muladd(p1, 1/p1, 1) == 2

        ρ = (2.7 ± 0.2)u"g/cm^3"
        mass = (250 ± 10)u"g"
        width = (30.5 ± 0.2)u"cm"
        l = (14.24 ± 0.2)u"m"
        thickness = u"μm"(mass/(ρ*width*l))
        @test thickness ≈ (21.3 ± 1.8)u"μm"

        @test ustrip(mass) ≈ 250 ± 10
        @test ustrip(mass) isa Particles

        a = (200 + 20*PT())u"ms"
        @test unit(a) == unit(1u"ms")

        b = ustrip(a)

        c = uconvert(u"s", a)
        @test c ≈ (0.200 + 0.020*PT())u"s"

        d1 = upreferred(a)
        @test d1 ≈ (0.200 + 0.020*PT())u"s"

        # Issue #152
        x = PT()
        z = 123u"m"
        y = z * x
        @test y isa PT{<:Quantity}
        @test ustrip(u"cm", x*y) isa PT{Float64}

        y = x * z
        @test y isa PT{<:Quantity}
        @test ustrip(u"cm", x*y) isa PT{Float64}
    end

end
