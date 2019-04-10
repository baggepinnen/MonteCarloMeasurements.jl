function systematic_sample(N, d=Normal(0,1); permute=true)
    e = rand()/N
    y = e:1/N:1
    ifun = invfun(d)
    par = params(d)
    o = map(y) do y
        ifun(par..., y)
    end
    if permute
        permute!(o, randperm(N))
    end
    o
end

struct SystematicNormal{T<:Real} <: UnivariateDistribution{T}
    μ::T
    σ::T
end
SystematicNormal() = SystematicNormal(0.,1.)

Base.rand(d::SystematicNormal, N) = d.μ .+ d.σ .* sysrandn(N)


invfun(::Normal) = StatsFuns.norminvcdf
invfun(::Gamma) = StatsFuns.gammainvcdf
invfun(::Poisson) = StatsFuns.poisinvcdf
invfun(::TDist) = StatsFuns.tdistinvcdf
invfun(::Beta) = StatsFuns.betainvcdf
invfun(x) = nothing
