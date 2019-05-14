using Test, MacroTools, MonteCarloMeasurements, ControlSystems
using MonteCarloMeasurements: ∓, ±

unsafe_comparisons()
# P = tf(1 ∓ 0.1, [1, 1∓0.1])
P = tf(1 ± 0.1, [1, 1±0.1])

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
build_result_container(P) = build_container(P, ) # TODO: we have no idea how to go from a vector of results to a single object with particles. Functions in general might turn an object into an arbitrarily different object (not only tf -> tf). One can perhaps analyze the result vector to see where it differes, but this feels ugly. Let the user provide the answer?
function build_container(P,condition=P->P isa AbstractParticles,result = P->P[1])
    # @show typeof(P)
    condition(P) && (return result(P)) # This replaces a Special with a float
    has_particles(P) || (return P) # No need to carry on
    P isa Number && (return P)
    if P isa AbstractArray # Special handling for arrays
        return map(P->build_container(P,condition,result), P)
    end
    fields = map(fieldnames(typeof(P))) do n
        f = getfield(P,n)
        has_particles(f) || (return f)
        # @show typeof(f), n
        build_container(f,condition,result)
    end
    T = nakedtypeof(P)

    try
        return T(fields...)
    catch e
        @error("Failed to create a `$T` by calling it with its fields in order. For this to work, `$T` must have a constructor that accepts all fields in the order they appear in the struct and accept that the fields that contained particles are replaced by 0. Try defining a meaningful constructor that accepts arguments with the type signature \n`$(T)$(typeof.(fields))`\nThe error thrown by `$T` was ")
        rethrow(e)
    end

end

particletype(p::AbstractParticles{T,N}) where {T,N} = (T,N)
particletype(::Type{<:AbstractParticles{T,N}}) where {T,N} = (T,N)

function particle_paths(P, allpaths=[], path=[])
    T = typeof(P)
    if T <: AbstractParticles
        push!(allpaths, (path,particletype(T)...))
        return
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


function get_setter_exprs(paths,P,P2)
    T,T2 = typeof(P), typeof(P2)
    exprs = map(paths) do p # for each encountered particle
        get1expr = :(P)
        set1expr = :(P2)
        for (i,(fn,ft,fs)) in enumerate(p[1][1:end-1]) # p[1] is a tuple vector where the first element in each tuple is the fieldname
            if ft <: AbstractArray
                # Here we should recursively branch down into all the elements, but this seems very complicated so we'll just access element 1 for now
                get1expr = :($(get1expr).$(fn)[1])
                set1expr = :($(set1expr).$(fn)[1])
            else # It was no array, regular field
                get1expr = :($(get1expr).$(fn))
                set1expr = :($(set1expr).$(fn))
            end
        end
        (fn,ft,fs) = p[1][end] # The last element, we've reached the particles
        if ft <: AbstractArray
            get1expr = :(getindex.($(get1expr).$(fn), partind)) # TODO: base.cartesian to avoid forming the vector from getindex?
            set1expr = :($(set1expr).$(fn) .= $get1expr) # TODO: not sure about the get1expr here

        else # It was no array, regular field
            get1expr = :($(get1expr).$(fn)[partind])
            set1expr = :($(get1expr).$(fn)[partind] = $get1expr)
        end
        get1expr, set1expr
    end
    get1expr, set1expr = getindex.(exprs, 1), getindex.(exprs, 2)
    set1expr = Expr(:block, set1expr...)
    get1expr = Expr(:block, get1expr...)
    # get2expr = MacroTools.postwalk(get1expr) do x
    #     x == :P && (return :P2)
    #     x == :P2 && (return :P)
    #     x
    # end
    set2expr = MacroTools.postwalk(set1expr) do x
        x == :P && (return :P2res)
        x == :P2 && (return :Pres)
        @capture(x, getindex.(y__,z_)) && (return :($(y)))
        @capture(x, y__ .= z_) && (return :(setindex!.($(y...), $(z...), partind)))
        x
    end
    # get1expr, set1expr, get2expr, set2expr
    set1expr, set2expr
end

##
# We create a two-stage process with the outer function `withbuffer` and an inner macro with the same name. The reason is that the function generates an expression at *runtime* and this should ideally be compiled into the body of the function without a runtime call to eval. The macro allows us to do this

# This seems to be required to splice expressions into the quote returned by a macro
macro eval2(ex)
    :($(ex))
end

macro withbuffer(f,P,P2,setters,setters2,N)
    quote
        $(esc(:(partind = 1))) # Because we need the actual name partind
        $(esc(@eval2(setters)))
        # $(esc(Pe)) = $(esc(P))
        $(esc(:P2res)) = $(esc(f))($(esc(P2))) # We first to index 1 to peek at the result
        $(esc(:Pres)) = deepcopy($(esc(P))) #build_result_container(paths, results[1])
        $(esc(@eval2(setters2)))
        for $(esc(:partind)) = 2:$(esc(N))
            $(esc(@eval2(setters)))
            # if $(esc(:partind)) <= 10
            #     display($(esc(P2)))
            #     display($(esc(:Pres)))
            # end
            $(esc(:P2res)) = $(esc(f))($(esc(P2)))
            $(esc(@eval2(setters2)))
        end
        $(esc(:Pres))
    end
end

function withbufffer(f,P)
    paths = particle_paths(P)
    P2 = build_container(P)
    setters, setters2 = get_setter_exprs(paths,P,P2)
    @assert all(n == paths[1][3] for n in getindex.(paths,3))
    N = paths[1][3]
    @withbuffer(f,P,P2,setters,setters2,N)
    # ex = @macroexpand @withbuffer(f,P,P2,setters,setters2,N)
    # display(prettify(ex))
end


using BenchmarkTools
P = tf(1 ± 0.1, [1, 1±0.1])
@btime code = withbufffer(P2->c2d(P2,0.1), P)
##

@btime foreach(i->c2d($(tf(1.,[1., 1])),0.1), 1:500)

res = @withbufffer P2->c2d(P2,0.1) P
##

paths = particle_paths(P)
P2 = build_container(P)
setters, setters2 = get_setter_exprs(paths,P,P2)
@assert all(n == paths[1][3] for n in getindex.(paths,3))
N = paths[1][3]
f = P2->c2d(P2,0.1)
ex = @macroexpand @withbuffer(f,P,P2,setters,setters2,N)
prettify(ex) |> display
##


@time eval(code)
@btime c2d($(tf(1,[1,1])),0.1)


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
