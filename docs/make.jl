using Documenter, MonteCarloMeasurements

makedocs(doctest = false) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
    repo = "github.com/baggepinnen/MonteCarloMeasurements.jl.git",
    julia  = "1.0",
    osname = "linux"
)
