using MonteCarloMeasurements, ForwardDiff, MacroTools
using MonteCarloMeasurements: ±
import ForwardDiff: Dual, gradient

const c = 1 ± 0.1 # These are the uncertain parameters
const d = 1 ± 0.1 # These are the uncertain parameters
# In the cost function below, we ensure that $cx+dy > 10 \; ∀ \; c,d ∈ P$ by looking at the worst case
function cost(params)
    x,y = params
    -(3x+2y) + 10000sum(params .< 0) + 10000*(maximum(c*x+d*y) > 10)
end

params = [1., 2] # Initial guess
paramsp = [1., 2] .± 0.001 # Initial guess
cost(params)     # Try the cost function

macro andreverse(ex)
    def = splitdef(ex)
    if haskey(def,:whereparams) && !isempty(def[:whereparams])
        quote
            $(esc(ex))
            $(esc(def[:name]))($(esc(def[:args][2])), $(esc(def[:args][1]))) where $(esc(def[:whereparams]...)) = $(esc(def[:body]))
        end
    else
        quote
            $(esc(ex))
            $(esc(def[:name]))($(esc(def[:args][2])), $(esc(def[:args][1]))) = $(esc(def[:body]))
        end
    end
end

display(@macroexpand(@andreverse Base.:(*)(p::PT, d::Dual{T}) where T = PT(p.particles .* Ref(d)))  |> prettify)

for PT in (Particles, StaticParticles)
    @eval begin
        @andreverse Base.:(*)(p::$PT, d::Dual) = $PT(p.particles .* Ref(d)) # To avoid ambiguity
        @andreverse Base.:(+)(p::$PT, d::Dual) = $PT(p.particles .+ Ref(d))
        # Base.promote_rule(::Type{Dual{S,F,D}}, ::Type{$PT{T,N}}) where {S,F,D,T,N} = $PT{Dual{S,promote_type(F,T),D},N}
        function ForwardDiff.extract_gradient!(::Type{T}, result::AbstractArray, dual::$PT{V,N}) where {T,V,N}
            d = ForwardDiff.partials.(T, dual.particles)

            copyto!(result, Particles(copy(reduce(hcat, d)')))
        end
    end
end

ForwardDiff.valtype(p::Particles{T,N}) where {T,N} = Particles{ForwardDiff.valtype(T), N}

Base.hidigit(x::AbstractParticles, base) = Base.hidigit(mean(x), base) # To avoid stackoverflow in some printing situations
Base.hidigit(x::Dual, base) = Base.hidigit(x.value, base) # To avoid stackoverflow in some printing situations
Base.round(d::Dual, r::RoundingMode) = round(d.value,r)


# c.f.
# extract_gradient!(::Type{T}, result::AbstractArray, y::Real) where {T} = fill!(result, zero(y))
# extract_gradient!(::Type{T}, result::AbstractArray, dual::Dual) where {T}= copyto!(result, partials(T, dual))

gradient(cost, params) == @unsafe gradient(cost, paramsp)

gradient(sum, params) == gradient(sum, paramsp)

all(gradient(prod, params) .≈ gradient(prod, paramsp))

unsafe_comparisons(false)
gradient(x -> params'x, params)
gradient(x -> paramsp'x, params)
gradient(x -> params'x, paramsp)
a = gradient(x -> paramsp'x, paramsp)

function strange(x,y)
    # T = promote_type(eltype(x), eltype(y))
    sum(x.^2+y.^2)
end

gradient(x->strange(x,params), params)
gradient(x->strange(x,params), paramsp) # This would be nice to support
gradient(x->strange(x,paramsp), params) # This use case is robust_opt
gradient(x->strange(x,paramsp), paramsp) # maybe this is a bit overkill
