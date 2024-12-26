# functions and docstring are here, but the actual implementation is in ext/RecipesBaseExt.jl

"""
    errorbarplot(x,y,[q=0.025])

Plots a vector of particles with error bars at quantile `q`.
If `q::Tuple`, then you can specify both lower and upper quantile, e.g., `(0.01, 0.99)`.
"""
function errorbarplot end

"""
    mcplot(x,y,[N=0])

Plots all trajectories represented by a vector of particles. `N > 1` controls the number of trajectories to plot.
"""
function mcplot end

"""
    ribbonplot(x,y,[q=0.025]; N=true)

Plots a vector of particles with a ribbon covering quantiles `q, 1-q`.
If `q::Tuple`, then you can specify both lower and upper quantile, e.g., `(0.01, 0.99)`.

If a positive number `N` is provided, `N` sample trajectories will be plotted on top of the ribbon.
"""
function ribbonplot end
