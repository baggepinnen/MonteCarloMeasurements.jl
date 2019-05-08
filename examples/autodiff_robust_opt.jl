# In this example we solve a robust linear programming problem using Optim. The problem is taken from [wikipedia](https://en.wikipedia.org/wiki/Robust_optimization#Example_1)

# $$\text{maximize}_{x,y} \; 3x+2y \quad \text{s.t}. x,y > 0, \quad cx+dy < 10 ∀ c,d ∈ P$$

# Where $c$ and $d$ are uncertain. We encode the constraint into the cost and solve it using 4 different algorithms


using MonteCarloMeasurements, Optim, ForwardDiff
using MonteCarloMeasurements: ∓

const c = 1 ∓ 0.1 # These are the uncertain parameters
const d = 1 ∓ 0.1 # These are the uncertain parameters
# In the cost function below, we ensure that $cx+dy > 10 \; ∀ \; c,d ∈ P$ by looking at the worst case
function cost(params)
    x,y = params
    -(3x+2y) + 10000sum(params .< 0) + 10000*(maximum(c*x+d*y) > 10)
end

params = [1., 1] # Initial guess
cost(params)     # Try the cost function

# If we do not define this method, we'll get a method ambiguity error
Base.:(*)(p::StaticParticles, d::ForwardDiff.Dual) = StaticParticles(p.particles .* Ref(d))
# We now solve the problem using the following list of algorithms
function solvemany()
    algos = [NelderMead(), SimulatedAnnealing(), BFGS(), Newton()]
    map(algos) do algo
        res = Optim.optimize(cost, params, algo, autodiff=:forward)
        m = res.minimizer
        cost(m)
    end
end
solvemany()'

# All methods find more or less the same minimum, but the gradient-free methods actually do a bit better
# How long time does it take to solve all the problems?
@time solvemany();
# It was quite fast
