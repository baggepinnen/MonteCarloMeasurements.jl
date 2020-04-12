using Unitful
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

    for PT in (Particles, StaticParticles)
        p1 = PT(100, Uniform(-0.5,1.5)) * 1u"V"
        p2 = PT(100, Uniform(-0.5,1.5)) * u"V"
        @test_nowarn println(p1)
        @test_nowarn display(p1)

        @test typeof(p1) == typeof(p2)

        p3 = unitful_testfunction(p1)
        @test extrema(p3) == (0.0u"V", 1.0u"V")
    end

end
