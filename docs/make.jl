using Documenter, MonteCarloMeasurements

makedocs(sitename="MonteCarloMeasurements Documentation", doctest = false, modules=[MonteCarloMeasurements],
  pages = [
        "Home" => "index.md",
        "Supporting new functions" => "overloading.md",
        "Examples" => "examples.md",
        "API" => "api.md"]
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
    repo = "github.com/baggepinnen/MonteCarloMeasurements.jl.git"
)
