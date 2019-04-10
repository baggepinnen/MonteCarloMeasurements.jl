function sysrandn(N;permute=false)
    e = rand()/N
    y = e:1/N:1
    o = StatsFuns.norminvcdf.(y)
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
