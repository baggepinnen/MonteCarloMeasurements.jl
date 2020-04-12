
@recipe function plot(p::AbstractParticles)
    seriestype --> :histogram
    @series p.particles
end

# @recipe f(::Type{<:AbstractParticles}, p::AbstractParticles) = p.particles # Does not seem to be needed


const RealOrTuple = Union{Real, Tuple}
handle_args(y::MvParticles, q::RealOrTuple=0.025) = 1:length(y), y, q
handle_args(x::AbstractVector, y::MvParticles, q::RealOrTuple=0.025) = x, y, q
handle_args(p) = handle_args(p.args...)
handle_args(args...) = throw(ArgumentError("The plot function should be called with the signature plotfun([x=1:length(y)], y::Vector{Particles}, [q=0.025])"))

function quantiles(y,q::Number)
    m = mean.(y)
    q > 0.5 && (q = 1-q)
    lower = -(quantile.(y,q)-m)
    upper = quantile.(y,1-q)-m
    lower,upper
end

function quantiles(y,q)
    m = mean.(y)
    lower = -(quantile.(y,q[1])-m)
    upper = quantile.(y,q[2])-m
    lower,upper
end

@userplot Errorbarplot
@recipe function plt(p::Errorbarplot)
    x,y,q = handle_args(p)
    m = mean.(y)
    label --> "Mean with $q quantile"
    yerror := quantiles(y, q)
    x,m
end

@userplot MCplot
@recipe function plt(p::MCplot)
    x,y,q = handle_args(p)
    label --> ""
    alpha --> 1/log(nparticles(y))
    m = Matrix(y)'
    x,m
end

@userplot Ribbonplot
@recipe function plt(p::Ribbonplot)
    x,y,q = handle_args(p)
    label --> "Mean with $q quantile"
    m = mean.(y)
    ribbon := quantiles(y, q)
    x,m
end

"""
errorbarplot(x,y,[q=0.025])

Plots a vector of particles with error bars at quantile `q`.
If `q::Tuple`, then you can specify both lower and upper quantile, e.g., `(0.01, 0.99)`.
"""
errorbarplot

"""
mcplot(x,y,[q=0.025])

Plots all trajectories represented by a vector of particles
"""
mcplot

"""
ribbonplot(x,y,[q=0.025])

Plots a vector of particles with a ribbon covering quantiles `q, 1-q`.
If `q::Tuple`, then you can specify both lower and upper quantile, e.g., `(0.01, 0.99)`.
"""
ribbonplot



@recipe function plt(y::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, q=0.025)
    label --> "Mean with ($q, $(1-q)) quantiles"
    ribbon := quantiles(y, q)
    mean.(y)
end


@recipe function plt(func::Function, x::MvParticles, q=0.025)
    y = func.(x)
    label --> "Mean with ($q, $(1-q)) quantiles"
    xerror := quantiles(x, q)
    yerror := quantiles(y, q)
    mean.(x), mean.(y)
end

@recipe function plt(x::MvParticles, y::MvParticles, q=0.025; points=false)
    my = mean.(y)
    mx = mean.(x)
    if points
        @series begin
            seriestype --> :scatter
            primary := false
            alpha --> 0.1
            Matrix(x), Matrix(y)
        end
    else
        yerror := quantiles(y, q)
        xerror := quantiles(x, q)
        label --> "Mean with $q quantile"
    end
    mx, my
end

@recipe function plt(x::MvParticles, y::AbstractArray, q=0.025)
    mx = mean.(x)
    lower,upper = quantiles(x, q)
    xerror := (lower,upper)
    mx, y
end

@recipe function plt(x::AbstractArray, y::MvParticles, q=0.025)
    ribbon := quantiles(y, q)
    label --> "Mean with ($q, $(1-q)) quantiles"
    x, mean.(y)
end
