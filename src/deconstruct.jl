using Test, MacroTools, MonteCarloMeasurements, ControlSystems, Polynomials
using MonteCarloMeasurements: ∓

unsafe_comparisons()
P = tf(1 ∓ 0.1, [1, 1∓0.1])

function has_particles(P)
    P isa AbstractParticles && (return true)
    P isa AbstractArray && (return has_particles(P[1]))
    any(fieldnames(typeof(P))) do n
        fp = getfield(P,n)
        if fp isa Union{AbstractArray, Tuple}
            return has_particles(fp[1]) # Specials can occur inside arrays or tuples
        else
            return has_particles(fp)
        end
    end
end

nakedtypeof(x::Type) = x.name.wrapper
nakedtypeof(x) = nakedtypeof(typeof(x))

function build_container(P,i=1)
    # @show typeof(P)
    P isa AbstractParticles && (return P[i]) # This replaces a Special with a float
    has_particles(P) || (return P) # No need to carry on
    P isa Number && (return P)
    if P isa AbstractArray # Special handling for arrays
        return map(P->build_container(P,i), P)
    end
    fields = map(fieldnames(typeof(P))) do n
        f = getfield(P,n)
        has_particles(f) || (return f)
        # @show typeof(f), n
        build_container(f,i)
    end
    T = nakedtypeof(P)

    try
        return T(fields...)
    catch e
        @error("Failed to create a `$T` by calling it with its fields in order. For this to work, `$T` must have a constructor that accepts all fields in the order they appear in the struct and accept that the fields that contained particles are replaced by 0. Try defining a meaningful constructor that accepts arguments with the type signature \n`$(T)$(typeof.(fields))`\nThe error thrown by `$T` was ")
        rethrow(e)
    end

end

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

a = walk_down(typeof(P))

@test a == [ Any[(:matrix, true), :num, (:a, true)], Any[(:matrix, true), :den, (:a, true)]]

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

paths = walk_down(P)

function get_accessors(paths,P,P2)
    T,T2 = typeof(P), typeof(P2)
    exprs = map(paths) do p # for each encountered particle
        getexpr = :(P)
        setexpr = :(P2)
        for (i,(fn,ft,fs)) in enumerate(p[1][1:end-1]) # p[1] is a tuple vector where the first element in each tuple is the fieldname
            if ft <: AbstractArray
                # Here we should recursively branch down into all the elements, but this seems very complicated so we'll just access element 1 for now
                getexpr = :($(getexpr).$(fn)[1])
                setexpr = :($(setexpr).$(fn)[1])
            else # It was no array, regular field
                getexpr = :($(getexpr).$(fn))
                setexpr = :($(setexpr).$(fn))
            end
        end
        (fn,ft,fs) = p[1][end] # The last element, we've reached the particles
        if ft <: AbstractArray
            getexpr = :(getindex.($(getexpr).$(fn), i))
            setexpr = :($(setexpr).$(fn) .= $getexpr) # TODO: not sure about the getexpr here

        else # It was no array, regular field
            getexpr = :($(getexpr).$(fn)) # TODO: handle if this is the last expression, i.e., we've reached the particles. Break i == length(p[1]) out of the loop
            setexpr = :($(getexpr).$(fn))
        end
        getexpr, setexpr
    end
    getindex.(exprs, 1), getindex.(exprs, 2)
end

accessors, setters = get_accessors(paths,P,P2)
@test length(accessors) == 2
# @test accessors[1] == :(P.matrix[1].num.a)

using ControlSystems
ControlSystems.TransferFunction(matrix::Array{ControlSystems.SisoRational{Float64},2}, Ts::Float64, ::Int64, ::Int64) = TransferFunction(matrix,Ts)


@test nakedtypeof(P) == TransferFunction
@test nakedtypeof(typeof(P)) == TransferFunction
@test typeof(P) == TransferFunction{ControlSystems.SisoRational{StaticParticles{Float64,100}}}
P2 = build_container(P)
@test typeof(P2) == TransferFunction{ControlSystems.SisoRational{Float64}}
@test has_particles(P)
@test has_particles(P.matrix)
@test has_particles(P.matrix[1])
@test has_particles(P.matrix[1].num)
@test has_particles(P.matrix[1].num.a)
@test has_particles(P.matrix[1].num.a[1])
