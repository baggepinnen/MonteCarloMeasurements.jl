# In this example we solve a robust linear programming problem using Optim. The problem is taken from [wikipedia](https://en.wikipedia.org/wiki/Robust_optimization#Example_1)

# $$\text{maximize}_{x,y} \; 3x+2y \quad \text{s.t}. x,y > 0, \quad cx+dy < 10 ∀ c,d ∈ P$$

# Where $c$ and $d$ are uncertain. We encode the constraint into the cost and solve it using 4 different algorithms


using MonteCarloMeasurements, Optim, ForwardDiff, Zygote

const c = 1 ∓ 0.1 # These are the uncertain parameters
const d = 1 ∓ 0.1 # These are the uncertain parameters
# In the cost function below, we ensure that $cx+dy > 10 \; ∀ \; c,d ∈ P$ by looking at the worst case
Base.findmax(p::AbstractParticles;dims=:) = findmax(p.particles,dims=:)
function cost(pars)
    x,y = pars
    -(3x+2y) + 10000sum(pars .< 0) + 10000*(maximum(c*x+d*y) > 10)
end

pars = [1., 1] # Initial guess
cost(pars)     # Try the cost function
cost'(pars)
# We now solve the problem using the following list of algorithms
function solvemany()
    algos = [NelderMead(), SimulatedAnnealing(), BFGS(), Newton()]
    map(algos) do algo
        res = Optim.optimize(cost, p->cost'(p), pars, algo, inplace=false)
        m = res.minimizer
        cost(m)
    end
end
solvemany()'

# All methods find more or less the same minimum, but the gradient-free methods actually do a bit better
# How long time does it take to solve all the problems?
@time solvemany();
# It was quite fast

# We can also see whether or not it's possible to take the gradient of
# 1. An deterministic function with respect to determinisitc parameters
# 2. An deterministic function with respect to uncertain parameters
# 3. An uncertain function with respect to determinisitc parameters
# 4. An uncertain function with respect to uncertain parameters
function strange(x,y)
    (x.^2)'*(y.^2)
end
deterministic = [1., 2] # Initial guess
uncertain = [1., 2] .+ 0.001 .* StaticParticles.(10) # Initial guess
ForwardDiff.gradient(x->strange(x,deterministic), deterministic)
#
ForwardDiff.gradient(x->strange(x,deterministic), uncertain)
#
ForwardDiff.gradient(x->strange(x,uncertain), deterministic)
#
a = ForwardDiff.gradient(x->strange(x,uncertain), uncertain);
# mean.(a)
# The last one here is commented because it sometimes segfaults. When it doesn't, it seems to produce the correct result with the complicated type Particles{Particles{Float64,N},N}, which errors when printed.

# We can also do the same using Zygote. The result is the same, and Zygote also handles the last version without producing a weird type in the result!
using Zygote
Zygote.gradient(x->strange(x,deterministic), deterministic)
#
Zygote.gradient(x->strange(x,deterministic), uncertain)
#
Zygote.gradient(x->strange(x,uncertain), deterministic)
#
Zygote.gradient(x->strange(x,uncertain), uncertain)
