using BSON, MonteCarloMeasurements, ControlSystems, Polynomials
cd("/local/home/fredrikb/robust_opt")
BSON.@load "P2.bson" P2
using MacroTools, Test

T = typeof(P2)

function walk_down(T::Type, allpaths=[], path=[])
    if T <: AbstractParticles
        push!(allpaths, path)
        return
    end
    for n in fieldnames(T)
        FT = fieldtype(T,n)
        if FT <: Union{AbstractArray, Tuple}
            walk_down(eltype(FT), allpaths, [path; (n, true)])
        else
            walk_down(FT, allpaths, [path; n])
        end
    end
    allpaths
end

a = walk_down(T)

@test a == [ Any[(:matrix, true), :num, (:a, true)], Any[(:matrix, true), :den, (:a, true)]]


function has_particles(P)
    P isa AbstractParticles && (return true)
    any(fieldnames(typeof(P))) do n
        fp = getfield(P,n)
        if fp isa Union{AbstractArray, Tuple}
            return has_particles(fp[1])
        else
            return has_particles(fp)
        end
    end
end

@test has_particles(P2)
@test !has_particles([1,2])

particletype(p::AbstractParticles{T,N}) where {T,N} = (T,N)
particletype(::Type{<:AbstractParticles{T,N}}) where {T,N} = (T,N)

function walk_down(P, allpaths=[], path=[])
    T = typeof(P)
    if T <: AbstractParticles
        push!(allpaths, (path,particletype(T)...))
        return
    end
    for n in fieldnames(T)
        fp = getfield(P,n)
        FT = typeof(fp)
        if FT <: Union{AbstractArray, Tuple}
            walk_down(fp[1], allpaths, [path; (n, FT, size(fp))])
        else
            walk_down(fp, allpaths, [path; (n, FT, ())])
        end
    end
    allpaths
end

paths = walk_down(P2)


con(p::TransferFunction, fields...) = TransferFunction(fields...)

function build_container(P)
    P isa AbstractParticles && (return zero(particletype(P)))
    has_particles(P) || (return P) # No need to carry on
    P isa Number && (return P)
    if P isa AbstractArray
        return map(P) do p
            build_container(p)
        end
    end
    fns = [fieldnames(typeof(P))...]
    fields = map(fns) do n
        f = getfield(P,n)
        has_particles(f) || (return f)
        build_container(f)
    end
    con(P,fields...)
end

a = build_container(P2)
# end


nested(T::Type, expr_builder, expr_combiner=default_combiner) =
    nested(T, Nothing, expr_builder, expr_combiner)
nested(T::Type, P::Type, expr_builder, expr_combiner) =
    expr_combiner(T, [Expr(:..., expr_builder(T, fn)) for fn in fieldnames(T)])

default_combiner(T, expressions) = Expr(:tuple, expressions...)

flatten_expr(T, path, x) = :(flatten(getfield($path, $(QuoteNode(x)))))
flatten(x::Number) = (x,)
flatten(x::AbstractParticles) = (zero(particletype(x)),)
flatten_inner(T) = nested(T, :t, flatten_expr) # Separated for inspectng code generation
@generated flatten(t) = flatten_inner(t)

flatten(P2)

flatten((1,2))
