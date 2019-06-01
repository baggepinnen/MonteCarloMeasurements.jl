struct Particles
    particles::Vector{Float64}
end

p = Particles(randn(10))

Base.:(^)(p::Particles,r) = Particles(p.particles.^r)
Base.:(>)(p::Particles, r) = Particles(map(>, p.particles, r))

function negsquare(x)
    x > 0 ? x^2 : -x^2
end
function negsquare2(x)
    y = x+1
    y += x > 0 ? x^2 : -x^2
    y
end

function negsquare3(x)
    y = x+1
    y += Particles(Base.Cartesian.@ntuple 2 i->x[i] > 0 ? x[i]^2 : -x[i]^2)
    y
end


# function negsquare(x::Complicated)
#     if x.a > 0
#         return x.a^2
#     else
#         return -x.a^2
#     end
# end

##
# register_primitive(negsquare)
negsquare(p)

## Cassette
using Cassette, IRTools
Cassette.@context Ctx;
Cassette.prehook(::Ctx, f::typeof(>), args...) = nothing # To reset
function Cassette.prehook(ctx::Ctx, f::typeof(>), p::Particles, r)
    @show ctx
    @show ctx.metadata
    @show p
    @show f.(p.particles,r)
    # Cassette.recurse(ctx, f, p, r)
end

# Cassette.overdub(Ctx(), negsquare, p)
# Cassette.overdub(Ctx(), negsquare, Complicated(p))

# Some use cases, however, require the ability to access and/or alter properties of the execution trace that just can't be reached via simple method overloading, like control flow or the surrounding scope of a method call. In these cases, you probably do want to manually implement a compiler pass! To facilitate these use cases, Cassette allows users to write and inject their own arbitrary NOTE: **post-lowering, pre-inference** compiler passes as part of the overdubbing process.
##
code = Meta.@lower x > 0 ? x^2 : -x^2
f(x) = x > 0 ? x^2 : -x^2
code_ = @code_lowered f(1.)
code__ = @code_typed f(1.)

code2 = Meta.@lower map(x.particles) do x
    x > 0 ? x^2 : -x^2
end

code3 = Meta.@lower begin
    y = similar(x)
    for i in 1:length(x)
        y[i] = x[i] > 0 ? x[i]^2 : -x[i]^2
    end
end

code4 = Meta.@lower begin
    ntuple(i->x[i] > 0 ? x[i]^2 : -x[i]^2)
end

code5 = Meta.@lower Particles(Base.Cartesian.@ntuple 2 i->x[i] > 0 ? x[i]^2 : -x[i]^2)

##
using IRTools
using IRTools: IR, var, block, blocks, @dynamo
using MacroTools

ir = IR()
x = IRTools.argument!(ir,2)
x2 = push!(ir, :($x*$x))
push!(ir, :(3*$x2 + 2*$x + 1))
ir[var(2)]


block(code5.args[1], 1)

@dynamo function mapifdyn(meta...)
    ir = IR(meta...)
    ir == nothing && return
    # pushfirst!(ir, :(println("hej")))
    for b in blocks(ir)
        @show b
    end

    ir
end
IRTools.refresh()
mapifdyn(negsquare3, p)


figure out which ssa are particles by looking at meta
walk ir until particles appears in branch
replicate branch like described below
make sure ssa is updated

ir = @code_ir negsquare(p)
bs = IRTools.blocks(ir)


# first there is a getindex
# then the original comparison
# original goto with adjusted numbers
# all instances of x are replaced by a reference to a variable resulting from getindex
# There are two getindex calls for each index, this can probably be reduced to a single variable
# all final goto jump the same number of steps
# all gotoifnot ahve the same distance between target and condition ssa
# branch body looks identical (if extra getindex removed) except for reference to ssa instead of to x directly

##
# The compiler pass needs to
# - Change every time a Particles appear in a boolean context to a function call
if p::Particles{Bool}
    f(p, args...)
end
# to
map(p.particles) do pi
    if pi
        f(pi, args...)
    end
end

# insert_statements!(code::Vector, codelocs::Vector, stmtcount, newstmts)
# For every statement stmt at position i in code for which stmtcount(stmt, i) returns an Int, remove stmt, and in its place, insert the statements returned by newstmts(stmt, i). If stmtcount(stmt, i) returns nothing, leave stmt alone.
# - stmtcount and newstmts must obey stmtcount(stmt, i) == length(newstmts(stmt, i)) if isa(stmtcount(stmt, i), Int).

# pass = Cassette.@pass transform # transform must be a Julia object that is callable with the following signature:
# transform(::Type{<:Context}, ::Cassette.Reflection)::Union{Expr,CodeInfo}
# Cassette provides a few IR-munging utility functions of interest to pass authors; for details, see insert_statements!, replace_match!, and is_ir_element.

# replace_match!(replace, ismatch, x)
# Return x with all subelements y replaced with replace(y) if ismatch(y). If !ismatch(y), but y is of type Expr, Array, or SubArray, then replace y in x with replace_match!(replace, ismatch, y).
##
using Cassette
Cassette.@context Ctx;

Base.:(>)(p::Particles, r) = Particles(map(>, p.particles, r))

newstmts = function (stmt, i)
    # [Expr(:(=), callbackslot, getmetadata), stmt]
    # @show typeof(stmt)
    # @show stmt
    return [:(println($stmt)), stmt]# Return the new statements, the old statement is not kept by default
    # return []
end

"return nothing if nothing to be done. Otherwise return an Int. This int indicates how many new statements will be returned by newstmts. Keep these as anonymous functions, otherwise changes are not updated."
stmtcount = function (stmt, i) # i is the location of the statement, for some reason dangerous to call it stmtcount, it will never get called
    # @show i
    # i == 1 ? nothing : 2
    if stmt isa Expr #|| Cassette.is_ir_element(stmt)
        if stmt.head == :gotoifnot
            # @show stmt.args[1]
            return nothing
        end
    end
    nothing
    # 2 # Return the number of new statements, the old statement is not kept by default
    # nothing
end

contains_branch(ir::Core.CodeInfo) = any(contains_branch, ir.code)
contains_branch(ir::IR) = any(contains_branch, blocks(ir))
contains_branch(b::IRTools.BasicBlock) = !isempty(b.branches)
contains_branch(ex::Expr) = Base.Meta.isexpr(ex, :gotoifnot)
contains_branch(any) = false

function branch_indices(ir)
    res = map(enumerate(ir.code)) do (i,c)
        c.head == :gotoifnot ? i : 0
    end
    res = filter(x->x>0, res)
    isempty(res) ? nothing : res
end

branch_target(ex) = ex.args[2]

function mapif(::Type{<:Ctx}, reflection::Cassette.Reflection)
    ir = reflection.code_info
    any(x-> x <: Particles, reflection.signature.parameters) || (return ir) # No particles included in this call
    contains_branch(ir) || (return ir)
    branch_inds = branch_indices(ir) # Keep just to verify assert below

    stmtcount = function (stmt, i)
        contains_branch(stmt) || (return nothing)
        @assert i ∈ branch_inds
        return 1 # One function call replaces the branch
    end

    newstmts = function (stmt, i)
        @show branch_body = ir.code[i:branch_target(stmt)-1] # the branch  starts at the index of the gotoifnot and ends one before the branch target
        call = Expr(:call)
        [stmt] # Must be a vector
    end


    Cassette.insert_statements!(ir.code, ir.codelocs, stmtcount, newstmts) # It's good to send in the entire ir.code so that all SSAValues are updated
    ir # Must return ir
end


mapifpass = Cassette.@pass mapif# TODO: make const later
# const mapifpass2 = Cassette.@pass mapif

ctx = Ctx(pass=mapifpass)
Cassette.overdub(ctx, negsquare, p)



##
reflection = Cassette.reflect((typeof(negsquare),typeof(p)))
reflection.signature.parameters
# any(x-> x <: Particles, reflection.signature.parameters)
ir = reflection.code_info

contains_branch(ir)
branch_inds = branch_indices(ir)
branch = ir.code[branch_inds[1]]
branch.args
branch_target(branch)

branch_body = ir.code[2:branch_target(branch)]
code3 = Expr(:method, branch_body)
