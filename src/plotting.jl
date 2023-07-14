
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
    m = vec(pmean.(y))
    q > 0.5 && (q = 1-q)
    lower = reshape(-(pquantile.(vec(y),q)-m), size(y))
    upper = reshape(pquantile.(vec(y),1-q)-m, size(y))
    lower,upper
end

function quantiles(y,q)
    m = vec(pmean.(y))
    lower = reshape(-(pquantile.(vec(y),q[1])-m), size(y))
    upper = reshape(pquantile.(vec(y),q[2])-m, size(y))
    lower,upper
end

@userplot Errorbarplot
@recipe function plt(p::Errorbarplot; quantile=nothing)
    x,y,q = handle_args(p)
    q = quantile === nothing ? q : quantile
    m = pmean.(y)
    label --> "Mean with $q quantile"
    Q = quantiles(y, q)
    if y isa AbstractMatrix
        for c in 1:size(y,2)
            @series begin
                yerror := (Q[1][:,c], Q[2][:,c])
                x,m[:,c]
            end
        end
    else
        yerror := Q
        @series x,m
    end
end

"This is a helper function to make multiple series into one series separated by `Inf`. This makes plotting vastly more efficient."
function to1series(x,y)
    r,c = size(y)
    y2 = vec([y; fill(Inf, 1, c)])
    x2 = repeat([x; Inf], c)
    x2,y2
end

to1series(y) = to1series(1:size(y,1),y)

@userplot MCplot
@recipe function plt(p::MCplot)
    x,y,q = handle_args(p)
    N = nparticles(y)
    selected = q > 1 ? randperm(N)[1:q] : 1:N
    N = length(selected)
    label --> ""
    seriesalpha --> 1/log(N)
    if y isa AbstractMatrix
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
@recipe function plt(p::Ribbonplot; N=false, quantile=nothing)
    x,y,q = handle_args(p)
    q = quantile === nothing ? q : quantile
    if N > 0
        for col = 1:size(y,2)
            yc = y[:,col]
            m = pmean.(yc)
            @series begin
                label --> "Mean with $q quantile"
                ribbon := quantiles(yc, q)
                x,m
            end
            @series begin
                ribbon := quantiles(yc, q)
                m
            end
            @series begin
                M = Matrix(yc)
                np,ny = size(M)
                primary := false
                nc = N > 1 ? N : min(np, 50)
                seriesalpha --> max(1/sqrt(nc), 0.1)
                chosen = randperm(np)[1:nc]
                to1series(M[chosen, :]')
            end
        end
    else
        @series begin
            label --> "Mean with $q quantile"
            m = pmean.(y)
            ribbon := quantiles(y, q)
            x,m
        end
    end
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
    ribbonplot(x,y,[q=0.025]; N=true)

Plots a vector of particles with a ribbon covering quantiles `q, 1-q`.
If `q::Tuple`, then you can specify both lower and upper quantile, e.g., `(0.01, 0.99)`.

If a positive number `N` is provided, `N` sample trajectories will be plotted on top of the ribbon.
"""
ribbonplot



@recipe function plt(y::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, q=0.025; N=true, ri=true, quantile=nothing)
    q = quantile === nothing ? q : quantile
    label --> "Mean with ($q, $(1-q)) quantiles"
    for col = 1:size(y,2)
        yc = y[:,col]
        if ri
            @series begin
                ribbon := quantiles(yc, q)
                pmean.(yc)
            end
        end
        if N > 0
            @series begin
                M = Matrix(yc)
                np,ny = size(M)
                primary := !ri
                nc = N > 1 ? N : min(np, 50)
                seriesalpha --> max(1/sqrt(nc), 0.1)
                chosen = randperm(np)[1:nc]
                M[chosen, :]'
                # to1series(M[chosen, :]') # We actually want different columns to look different
            end
        end
    end
end


@recipe function plt(func::Function, x::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, q=0.025; quantile=nothing)
    y = func.(x)
    q = quantile === nothing ? q : quantile
    label --> "Mean with ($q, $(1-q)) quantiles"
    xerror := quantiles(x, q)
    yerror := quantiles(y, q)
    pmean.(x), pmean.(y)
end

@recipe function plt(x::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, y::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, q=0.025; points=false, quantile=nothing)
    my = pmean.(y)
    mx = pmean.(x)
    q = quantile === nothing ? q : quantile
    if points
        @series begin
            seriestype --> :scatter
            primary := true
            seriesalpha --> 0.1
            vec(Matrix(x)), vec(Matrix(y))
        end
    else
        @series begin
            yerror := quantiles(y, q)
            xerror := quantiles(x, q)
            label --> "Mean with $q quantile"
            mx, my
        end
    end
end

@recipe function plt(x::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, y::AbstractArray, q=0.025; quantile=nothing)
    mx = pmean.(x)
    q = quantile === nothing ? q : quantile
    lower,upper = quantiles(x, q)
    xerror := (lower,upper)
    mx, y
end

@recipe function plt(x::AbstractArray, y::Union{MvParticles,AbstractMatrix{<:AbstractParticles}}, q=0.025; N=true, ri=true, quantile=nothing)
    samedim = size(x) === size(y)
    layout --> max(size(x, 2), size(y, 2))
    q = quantile === nothing ? q : quantile
    if N > 0
        for col = 1:size(y,2)
            yc = y[:,col]
            if ri
                @series begin
                    seriescolor --> col
                    subplot --> col
                    ribbon := quantiles(yc, q)
                    label --> "Mean with ($q, $(1-q)) quantiles"
                    x, pmean.(yc)
                end
            end
            @series begin
                seriescolor --> col
                subplot --> col
                M = Matrix(yc)
                np,ny = size(M)
                primary := !ri
                nc = N > 1 ? N : min(np, 50)
                seriesalpha --> max(1/sqrt(nc), 0.1)
                chosen = randperm(np)[1:nc]
                to1series(samedim ? x[:, col] : x, M[chosen, :]')
            end
        end
    else
        @series begin
            ribbon := quantiles(y, q)
            label --> "Mean with ($q, $(1-q)) quantiles"
            x, pmean.(y)
        end
    end
end
