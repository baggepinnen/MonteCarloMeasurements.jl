
@recipe function plot(p::AbstractParticles)
    seriestype --> :histogram
    @series p.particles
end

# @recipe f(::Type{<:AbstractParticles}, p::AbstractParticles) = p.particles # Does not seem to be needed


const RealOrTuple = Union{Real, Tuple}
handle_args(y::AbstractVecOrMat{<:AbstractParticles}, q::RealOrTuple=0.025) = 1:size(y,1), y, q
handle_args(x::AbstractVector, y::AbstractVecOrMat{<:AbstractParticles}, q::RealOrTuple=0.025) = x, y, q
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

"This is a helper function to make multiple series into one series separated by `Inf`. This makes plotting vastly more efficient."
function to1series(x,y)
    r,c = size(y)
    y2 = vec([y; fill(Inf, 1, c)])
    x2 = repeat([x; Inf], c)
    x2,y2
end

@userplot MCplot
@recipe function plt(p::MCplot)
    x,y,q = handle_args(p)
    N = nparticles(y)
    selected = q > 1 ? randperm(N)[1:q] : 1:N
    N = length(selected)
    label --> ""
    alpha --> 1/log(N)
    if y isa Matrix
        for c in 1:size(y,2)
            m = Matrix(y[:,c])'
            @series to1series(x, m[:, selected])
        end
    else
        m = Matrix(y)'
        @series to1series(x, m[:, selected])
    end
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
mcplot(x,y,[N=0])

Plots all trajectories represented by a vector of particles. `N > 1` controls the number of trajectories to plot.
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
