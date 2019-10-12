"""
    systematic_sample(N, d=Normal(0,1); permute=true)
returns a `Vector` of length `N` sampled systematically from the distribution `d`. If `permute=false`, this vector will be sorted.
"""
function systematic_sample(N, d=Normal(0,1); permute=true)
    e   = 0.5/N # rand()/N
    y   = e:1/N:1
    o = map(y) do y
        quantile(d,y)
    end
    permute && permute!(o, randperm(N))
    o
end
