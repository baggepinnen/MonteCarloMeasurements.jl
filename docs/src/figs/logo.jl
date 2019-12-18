# PGFPlots.save("logo.tex", fig.o, include_preamble=true)
# add \begin{axis}[ticks=none,height...
# add begin{tikzpicture}[scale=0.5]
# Font: https://github.com/JuliaGraphics/julia-logo-graphics/blob/master/font/TamilMN-Bold.ttf
# pdftopng test.pdf logo.png
# convert logo.pdf logo.svg
using MonteCarloMeasurements, Plots, Random, KernelDensity, Colors
darker_blue = RGB(0.251, 0.388, 0.847)
lighter_blue = RGB(0.4, 0.51, 0.878)
darker_purple = RGB(0.584, 0.345, 0.698)
lighter_purple  = RGB(0.667, 0.475, 0.757)
darker_green  = RGB(0.22, 0.596, 0.149)
lighter_green  = RGB(0.376, 0.678, 0.318)
darker_red  = RGB(0.796, 0.235, 0.2)
lighter_red  = RGB(0.835, 0.388, 0.361)
N = 30
Random.seed!(3)
a,b,c = [0.2randn(N,2) for _ in 1:3]

a .+= [0 0]
b .+= [1 0]
c .+= [0.5 1]

pa,pb,pc = Particles.((a,b,c))

opts = (markersize=9, markerstrokewidth=2, markerstrokealpha=0.8, markeralpha=0.8, size=(300,300))
scatter(eachcol(a)...; c=lighter_red, markerstrokecolor=darker_red, opts..., axis=false, grid=false, legend=false)
scatter!(eachcol(b)...; c=lighter_purple, markerstrokecolor=darker_purple, opts...)
# scatter!(eachcol(c)...; c=lighter_green, markerstrokecolor=darker_, opts...)

ls = (linewidth=10,markerstrokewidth=2)
plot!(pa[1:1],pa[2:2]; c=lighter_red, markerstrokecolor=darker_red, ls...)
plot!(pb[1:1],pb[2:2]; c=lighter_purple, markerstrokecolor=darker_purple, ls...)
# plot!(pc[1:1],pc[2:2]; c=lighter_green, markerstrokecolor=darker_, ls...)

##
x = LinRange(-0.5,1.45,30)
f(x) = (x-0.3)^2 + 0.5
y = f.(x)
plot!(x,y, l=(darker_blue,))

bi = 1:3:N
plot!([b[bi,1] b[bi,1]]', [b[bi,2] f.(b[bi,1])]', l=(lighter_purple, :dash, 0.2))
plot!([b[bi,1] fill(0.5,length(bi))]', [f.(b[bi,1]) f.(b[bi,1])]', l=(lighter_green, :dash, 0.2))

scatter!(fill(0.5,N), f.(b[:,1]); c=lighter_green, markerstrokecolor=darker_green, opts...)

kd = kde(f.(b[:,1]), npoints=200, bandwidth=0.09, boundary=(0.5,1.8))
fig = plot!(0.5 .- 0.2kd.density, kd.x, c=lighter_green, markerstrokecolor=darker_green, fill=true, fillalpha=0.2)
# PGFPlots.save("test.tex", fig.o, include_preamble=true)
