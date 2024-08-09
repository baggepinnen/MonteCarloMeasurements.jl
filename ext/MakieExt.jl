module MakieExt

using Makie
using MonteCarloMeasurements

Makie.used_attributes(::Type{<:Series}, ::AbstractVector, ::AbstractVector{<:Particles}) = (:N,)
Makie.used_attributes(::Type{<:Series}, ::AbstractVector{<:Tuple{<:Real,<:Particles}}) = (:N,)
Makie.convert_arguments(ct::Type{<:Series}, x::AbstractVector, y::AbstractVector{<:Particles}; N=7) = convert_arguments(ct, x, Matrix(y)[1:min(N, end), :])

Makie.used_attributes(::Type{<:Union{Rangebars,Band}}, ::AbstractVector, ::AbstractVector{<:Particles}) = (:q, :nσ)
Makie.used_attributes(::Type{<:Union{Rangebars,Band}}, ::AbstractVector{<:Tuple{<:Real,<:Particles}}) = (:q, :nσ)
function Makie.convert_arguments(ct::Type{<:Union{Rangebars,Band}}, x::AbstractVector, y::AbstractVector{<:Particles}; q=nothing, nσ=nothing)
    !isnothing(q) && !isnothing(nσ) && throw(ArgumentError("Only one of `q`` or `nσ`` can be specified"))
    if isnothing(q) && isnothing(nσ)
        q = 0.16
    end
    if !isnothing(q)
        convert_arguments(ct, x, pquantile.(y, q), pquantile.(y, 1-q))
    elseif !isnothing(nσ)
        avgs = pmean.(y)
        Δs = nσ .* pstd.(y)
        convert_arguments(ct, x, avgs .- Δs, avgs .+ Δs)
    end
end

Makie.convert_arguments(ct::PointBased, x::AbstractVector{<:Real}, y::AbstractVector{<:Particles}; avgfunc=pmedian) = convert_arguments(ct, x, avgfunc.(y))

Makie.convert_arguments(ct::Type{<:AbstractPlot}, X::AbstractVector{<:Tuple{<:Real,<:Particles}}; kwargs...) = convert_arguments(ct, first.(X), last.(X); kwargs...)

Makie.convert_arguments(p::Type{<:Union{Hist,Density}}, x::Particles) = convert_arguments(p, Vector(x))

end
