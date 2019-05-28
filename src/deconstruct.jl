"""
    has_particles(P)
Determine whether or no the object `P` has some kind of particles inside it. This function examins fields of `P` recursively and looks inside arrays etc.
"""
function has_particles(P)
    P isa AbstractParticles && (return true)
    if P isa AbstractArray
        length(P) < 1 && (return eltype(P) <: AbstractParticles)
        return has_particles(P[1])
    end
    any(fieldnames(typeof(P))) do n
        fp = getfield(P,n)
        if fp isa Union{AbstractArray, Tuple}
            length(fp) < 1 && (return eltype(fp) <: AbstractParticles)
            return has_particles(fp[1]) # Specials can occur inside arrays or tuples
        else
            return has_particles(fp)
        end
    end
end

"""
    has_mutable_particles(P)
Similar to `has_particles`, but only returns true if the found particles are mutable, i.e., are not `StaticParticles`
"""
function has_mutable_particles(P)
    P isa Particles && (return true)
    P isa StaticParticles && (return false)
    P isa AbstractArray && (return has_mutable_particles(P[1]))
    all(fieldnames(typeof(P))) do n
        fp = getfield(P,n)
        if fp isa Union{AbstractArray, Tuple}
            length(fp) < 1 && (return eltype(fp) <: Particles)
            return has_mutable_particles(fp[1])
        else
            return has_mutable_particles(fp)
        end
    end
end
"""
    nakedtypeof(x)
Returns the type of `x` with type parameters removed. Uses internals and should ideally not be used at all. Do not use inside generated function.
"""
nakedtypeof(x::Type) = x.name.wrapper
nakedtypeof(x) = nakedtypeof(typeof(x))

"""
    build_mutable_container(P)
Recursively visits all fields of `P` and replaces all instances of `StaticParticles` with `Particles`
"""
function build_mutable_container(P)
    has_mutable_particles(P) && (return P)
    replace_particles(P, P->P isa AbstractParticles, P->Particles(Vector(P.particles)))
end

function make_scalar(P)
    replace_particles(P, P->P isa AbstractParticles, P->Particles([mean(P)]))
end

function restore_scalar(P, N)
    replace_particles(P, P->P isa AbstractParticles, P->Particles(N))
end

"""
    make_static(P)
Replaces all mutable particles inside `P` with `StaticParticles`.
"""
function make_static(P)
    !has_mutable_particles(P) && (return P)
    replace_particles(P, P->P isa AbstractParticles, P->StaticParticles(P.particles))
end

"""
    build_container(P)
Recursively visits all fields of `P` and replaces all instances of `AbstractParticles{T,N}` with `::T`
"""
build_container(P) = replace_particles(P,P->P isa AbstractParticles,P->P[1])

"""
    mean_object(x)
Returns an object similar to `x`, but where all internal instances of `Particles` are replaced with their mean. The generalization of this function is `replace_particles`.
"""
mean_object(p::AbstractParticles) = mean(p)
mean_object(p::AbstractArray{<:AbstractParticles}) = mean.(p)
mean_object(P) = replace_particles(P,P->P isa AbstractParticles,P->mean(P))

"""
    replace_particles(x,condition=P->P isa AbstractParticles,replacer = P->P[1])

This function recursively scans through the structure `x`, every time a field that matches `condition` is found, `replacer` is called on that field and the result is used instead of `P`. See function `mean_object`, which uses this function to replace all instances of `Particles` with their mean.
"""
function replace_particles(P,condition=P->P isa AbstractParticles,replacer = P->P[1])
    # @show typeof(P)
    condition(P) && (return replacer(P))
    has_particles(P) || (return P) # No need to carry on
    if P isa AbstractArray # Special handling for arrays
        return map(P->replace_particles(P,condition,replacer), P)
    end
    P isa Complex && condition(real(P)) && (return complex(replacer(real(P)), replacer(imag(P))))
    P isa Number && (return P)
    fields = map(fieldnames(typeof(P))) do n
        f = getfield(P,n)
        has_particles(f) || (return f)
        # @show typeof(f), n
        replace_particles(f,condition,replacer)
    end
    T = nakedtypeof(P)

    try
        return T(fields...)
    catch e
        if has_mutable_particles(P)
            @error("Failed to create a `$T` by calling it with its fields in order. For this to work, `$T` must have a constructor that accepts all fields in the order they appear in the struct and accept that the fields that contained particles are replaced by 0. Try defining a meaningful constructor that accepts arguments with the type signature \n`$(T)$(typeof.(fields))`\nThe error thrown by `$T` was ")
        else
            mutable_fields = build_mutable_container.(fields)
            @error("Failed to create a `$T` by calling it with its fields in order. For this to work, `$T` must have a constructor that accepts all fields in the order they appear in the struct and accept that the fields that contained particles are replaced by 0. Try defining a meaningful constructor that accepts arguments with the following two type signatures \n`$(T)$(typeof.(fields))`\n`$(T)$(typeof.(mutable_fields))`\nThe error thrown by `$T` was ")
        end
        rethrow(e)
    end

end

"""
particletype(p::AbstractParticles{T,N}) = (T,N)
"""
particletype(p::AbstractParticles{T,N}) where {T,N} = (T,N)
particletype(::Type{<:AbstractParticles{T,N}}) where {T,N} = (T,N)

"""
particle_paths(P)

Figure out all paths down through fields of `P` that lead to an instace of `<: AbstractParticles`. The returned structure is a list where each list element is a tuple. The tuple looks like this: (path, particletype, particlenumber)
`path in turn looks like this (:fieldname, fieldtype, size)
"""
function particle_paths(P, allpaths=[], path=[])
    T = typeof(P)
    if T <: AbstractParticles
        push!(allpaths, (path,particletype(T)...))
        return
    end
    if T <: Union{AbstractArray, Tuple}
        particle_paths(P[1], allpaths, [path; (:input, T, size(P))])
    end
    for n in fieldnames(T)
        fp = getfield(P,n)
        FT = typeof(fp)
        if FT <: Union{AbstractArray, Tuple}
            particle_paths(fp[1], allpaths, [path; (n, FT, size(fp))])
        else
            particle_paths(fp, allpaths, [path; (n, FT, ())])
        end
    end
    ntuple(i->allpaths[i], length(allpaths))
end

function vecpartind2vec!(v, pv, j)
    for i in eachindex(v)
        v[i] = pv[i][j]
    end
end

function vec2vecpartind!(pv, v, j)
    for i in eachindex(v)
        pv[i][j] = v[i]
    end
end

"""
    s1 = get_buffer_setter(paths)
Returns a function that is to be used to update work buffer inside `Workspace`
This function is `@eval`ed and can cause world-age problems unless called with `invokelatest`.
"""
function get_buffer_setter(paths)
    setbufex = map(paths) do p # for each encountered particle
        getbufex = :(input)
        setbufex = :(simple_input)
        for (i,(fn,ft,fs)) in enumerate(p[1][1:end-1]) # p[1] is a tuple vector where the first element in each tuple is the fieldname
            if ft <: AbstractArray
                # Here we should recursively branch down into all the elements, but this seems very complicated so we'll just access element 1 for now
                getbufex = :($(getbufex).$(fn)[1])
                setbufex = :($(setbufex).$(fn)[1])
            else # It was no array, regular field
                getbufex = :($(getbufex).$(fn))
                setbufex = :($(setbufex).$(fn))
            end
        end
        (fn,ft,fs) = p[1][end] # The last element, we've reached the particles
        if ft <: AbstractArray
            setbufex = :(vecpartind2vec!($(setbufex).$(fn), $(getbufex).$(fn), partind))
            # getbufex = :(getindex.($(getbufex).$(fn), partind))

        else # It was no array, regular field
            getbufex = :($(getbufex).$(fn)[partind])
            setbufex = :($(getbufex).$(fn)[partind] = $getbufex)
        end
        setbufex = MacroTools.postwalk(setbufex) do x
            @capture(x, y_.input) && (return y)
            x
        end
        setbufex
    end
    # setbufex,getbufex = getindex.(setbufex, 1),getindex.(getbufex, 2)
    setbufex = Expr(:block, setbufex...)
    # getbufex = Expr(:block, getbufex...)
    @eval setbuffun = (input,simple_input,partind)-> $setbufex
    setbuffun

end

function get_result_setter(result)
    paths = particle_paths(result)
    setresex = map(paths) do p # for each encountered particle
        getresex = :(result)
        setresex = :(simple_result)
        for (fn,ft,fs) in p[1][1:end-1] # p[1] is a tuple vector where the first element in each tuple is the fieldname
            if ft <: Union{AbstractArray, Tuple}
                # Here we should recursively branch down into all the elements, but this seems very complicated so we'll just access element 1 for now
                getresex = :($(getresex).$(fn)[1])
                setresex = :($(setresex).$(fn)[1])
            else # It was no array, regular field
                getresex = :($(getresex).$(fn))
                setresex = :($(setresex).$(fn))
            end
        end
        (fn,ft,fs) = p[1][end] # The last element, we've reached the particles
        if ft <: AbstractArray
            setresex = :(vec2vecpartind!($(getresex).$(fn), $(setresex).$(fn), partind))
            # getresex = :(getindex.($(getresex).$(fn), partind))

        else # It was no array, regular field
            getresex = :($(getresex).$(fn)[partind])
            setresex = :($(getresex).$(fn)[partind] = $getbufex)
        end
        setresex = MacroTools.postwalk(setresex) do x
            @capture(x, y_.input) && (return y)
            x
        end
        setresex
    end
    setresex = Expr(:block, setresex...)

    @eval setresfun = (result,simple_result,partind)-> $setresex
    setresfun
end

##
# We create a two-stage process with the outer function `withbuffer` and an inner macro with the same name. The reason is that the function generates an expression at *runtime* and this should ideally be compiled into the body of the function without a runtime call to eval. The macro allows us to do this

struct Workspace{T1,T2,T3,T4,T5,T6}
    simple_input::T1
    simple_result::T2
    result::T3
    buffersetter::T4
    resultsetter::T5
    f::T6
    N::Int
end

"""
    Workspace(f, input)
Create a `Workspace` object for inputs of type `typeof(input)`. Useful if `input` is a structure with fields of type `<: AbstractParticles` (can be deeply nested). See also `with_workspace`.
"""
function Workspace(f,input)
    paths = particle_paths(input)
    buffersetter = get_buffer_setter(paths)
    @assert all(n == paths[1][3] for n in getindex.(paths,3))
    simple_input = build_container(input)
    N = paths[1][3]
    Base.invokelatest(buffersetter,input,simple_input,1)
    simple_result = f(simple_input) # We first to index 1 to peek at the result
    result = @unsafe restore_scalar(build_mutable_container(f(make_scalar(input))), N) # Heuristic, see what the result is if called with particles and unsafe_comparisons TODO: If the reason the workspace approach is used is that the function f fails for different reason than comparinsons, this will fail here. Maybe Particles{1} can act as constant and be propagated through
    resultsetter = get_result_setter(result)
    Workspace(simple_input,simple_result,result,buffersetter, resultsetter,f,N)
end

"""
    with_workspace(f,P)

In some cases, defining a primitive function which particles are to be propagate through is not possible but allowing unsafe comparisons are not acceptable. One such case is functions that internally calculate eigenvalues of uncertain matrices. The eigenvalue calculation makes use of comparison operators. If the uncertainty is large, eigenvalues might change place in the sorted list of returned eigenvalues, completely ruining downstream computations. For this we recommend, in order of preference
1. Use `@bymap` detailed [in the readme](https://github.com/baggepinnen/MonteCarloMeasurements.jl#monte-carlo-simulation-by-mappmap). Applicable if all uncertain values appears as arguments to your entry function.
2. Create a `Workspace` object and call it using your entry function. Applicable if uncertain parameters appear nested in an object that is an argument to your entry function:
```julia
# desired computation: y = f(obj), obj contains uncertain parameters inside
y = with_workspace(f, obj)
# or equivalently
w = Workspace(obj)
use_invokelatest = true # Set this to false to gain 0.1-1 ms, at the expense of world-age problems if w is created and used in the same function.
w(f, use_invokelatest)
```
"""
with_workspace(f,P) = Workspace(f,P)(P, true)

function (w::Workspace)(input)
    simple_input,simple_result,result,buffersetter,resultsetter,N,f = w.simple_input,w.simple_result,w.result,w.buffersetter,w.resultsetter,w.N,w.f
    for partind = 1:N
        buffersetter(input,simple_input, partind)
        simple_result = f(simple_input)
        Base.invokelatest(resultsetter, result,simple_result, partind)
    end
    has_mutable_particles(input) ? result : make_static(result)
end

function (w::Workspace)(input, invlatest::Bool)
    invlatest || w(input)
    simple_input,simple_result,result,buffersetter,resultsetter,N,f = w.simple_input,w.simple_result,w.result,w.buffersetter,w.resultsetter,w.N,w.f

    for partind = 1:N
        Base.invokelatest(buffersetter, input,simple_input, partind)
        simple_result = f(simple_input)
        Base.invokelatest(resultsetter, result,simple_result, partind)
    end
    has_mutable_particles(input) ? result : make_static(result)
end

# macro withbuffer(f,P,simple_input,setters,setters2,N)
#     quote
#         $(esc(:(partind = 1))) # Because we need the actual name partind
#         $(esc(setters))($(esc(P)),$(esc(simple_input)), $(esc(:partind)))
#         $(esc(:simple_result)) = $(esc(f))($(esc(simple_input))) # We first to index 1 to peek at the result
#         result = @unsafe build_mutable_container($(esc(f))($(esc(P)))) # Heuristic, see what the result is if called with particles and unsafe_comparisons
#         $(esc(setters2))(result,$(esc(:simple_result)), $(esc(:partind)))
#         for $(esc(:partind)) = 2:$(esc(N))
#             $(esc(setters))($(esc(P)),$(esc(simple_input)), $(esc(:partind)))
#             $(esc(:simple_result)) = $(esc(f))($(esc(simple_input)))
#             $(esc(setters2))(result,$(esc(:simple_result)), $(esc(:partind)))
#         end
#         result
#     end
# end
#
# @generated function withbufferg(f,P,simple_input,N,setters,setters2)
#     ex = Expr(:block)
#     push!(ex.args, quote
#         partind = 1 # Because we need the actual name partind
#     end)
#     push!(ex.args, :(setters(P,simple_input,partind)))
#     push!(ex.args, quote
#         simple_result = f(simple_input) # We first to index 1 to peek at the result
#         result = @unsafe f(P)
#     end) #build_container(paths, results[1])
#     push!(ex.args, :(setters2(result,simple_result,partind)))
#     loopex = Expr(:block, :(setters(P,simple_input,partind)))
#     push!(loopex.args, :(simple_result = f(simple_input)))
#     push!(loopex.args, :(setters2(result,simple_result,partind)))
#     push!(ex.args, quote
#         for partind = 2:N
#             $loopex
#         end
#     end)
#     push!(ex.args, :result)
#     ex
# end
#
