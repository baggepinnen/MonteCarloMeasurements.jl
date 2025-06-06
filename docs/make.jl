using Documenter, MonteCarloMeasurements, Unitful

using Plots
plotly()

DocMeta.setdocmeta!(MonteCarloMeasurements, :DocTestSetup,
                    :(using MonteCarloMeasurements))

makedocs(
      sitename = "MonteCarloMeasurements Documentation",
      doctest = true,
      modules = [MonteCarloMeasurements],
      pages = [
            "Home" => "index.md",
            "Supporting new functions" => "overloading.md",
            "Examples" => "examples.md",
            "Linear vs. Monte-Carlo uncertainty propagation" => "comparison.md",
            "Performance tips" => "performance.md",
            "Advanced usage" => "advanced_usage.md",
            "API" => "api.md",
      ],
      format = Documenter.HTML(
            prettyurls = haskey(ENV, "CI"),
            sidebar_sitename = false,
      ),
      checkdocs = :export,
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
      #deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
      repo = "github.com/baggepinnen/MonteCarloMeasurements.jl.git",
)
