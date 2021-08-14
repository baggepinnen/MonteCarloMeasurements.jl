using MonteCarloMeasurements, ForwardDiff, Test
const FD = ForwardDiff

@testset "forwarddiff" begin
    @info "Testing ForwardDiff"
    c = 1 + 0.1Particles(500) # These are the uncertain parameters
    d = 1 + 0.1Particles(500) # These are the uncertain parameters
    # In the cost function below, we ensure that $cx+dy > 10 \; ∀ \; c,d ∈ P$ by looking at the worst case
    function cost(params)
        x,y = params
        -(3x+2y) + 10000sum(params .< 0) + 10000*(pmaximum(c*x+d*y) > 10)
    end

    params = [1., 2] # Initial guess
    paramsp = [1., 2] .+ 0.001 .* Particles(500) # Initial guess
    @test cost(params) == -7    # Try the cost function


    @test FD.gradient(cost, params) == @unsafe FD.gradient(cost, paramsp)
    @test FD.gradient(sum, params) == FD.gradient(sum, paramsp)
    @test all(FD.gradient(prod, params) .≈ FD.gradient(prod, paramsp))

    unsafe_comparisons(false)
    @test FD.gradient(x -> params'x, params) == params
    @test FD.gradient(x -> paramsp'x, params) == paramsp
    @test FD.gradient(x -> params'x, paramsp) == params
    r = FD.gradient(x -> paramsp'x, paramsp)
    @test pmean(pmean(r[1])) ≈ params[1] atol=1e-2
    @test pmean(pmean(r[2])) ≈ params[2] atol=1e-2

    @test FD.jacobian(x -> params+x, params) == I
    @test FD.jacobian(x -> paramsp+x, params) == I
    @test FD.jacobian(x -> params+x, paramsp) == I

    @test FD.jacobian(x -> params-x, params) == -I
    @test FD.jacobian(x -> paramsp-x, params) == -I
    @test FD.jacobian(x -> params-x, paramsp) == -I

    @test FD.jacobian(x -> x-params, params) == I
    @test FD.jacobian(x -> x-paramsp, params) == I
    @test FD.jacobian(x -> x-params, paramsp) == I

    function strange(x,y)
        (x.^2)'*(y.^2)
    end
    ref = FD.gradient(x->strange(x,params), params)
    @test FD.gradient(x->strange(x,params), paramsp) != ref
    @test FD.gradient(x->strange(x,params), paramsp) ≈ ref
    @test FD.gradient(x->strange(x,paramsp), params) != ref
    @test FD.gradient(x->strange(x,paramsp), params) ≈ ref
    r = FD.gradient(x->strange(x,paramsp), paramsp) # maybe this is a bit overkill
    @test pmean(pmean(r[1])) ≈ ref[1] atol=1e-2
    @test pmean(pmean(r[2])) ≈ ref[2] atol=1e-2
    @test pmean(pmean(r[1])) != ref[1]
    @test pmean(pmean(r[2])) != ref[2]

    unsafe_comparisons(false)
end
