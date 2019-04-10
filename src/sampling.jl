function sysrandn(N)
    e = rand()/N
    y = e:1/N:1
    StatsFuns.norminvcdf.(y)
end

struct SystematicNormal{T<:Real} <: UnivariateDistribution{T}
    μ::T
    σ::T
end
SystematicNormal() = SystematicNormal(0.,1.)

Base.rand(d::SystematicNormal, N) = d.μ .+ d.σ .* sysrandn(N)
