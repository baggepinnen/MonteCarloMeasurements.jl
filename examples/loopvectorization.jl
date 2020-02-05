using LoopVectorization
a = rand(Float32, 500)

@btime map(exp, a)
@btime vmap(exp, a)

@btime map(log, a)
@btime vmap(log, a)

@btime map(sin, a)
@btime vmap(sin, a)
