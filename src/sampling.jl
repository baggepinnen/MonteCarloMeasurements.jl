function systematic_sample(N, d=Normal(0,1); permute=true)
    e   = rand()/N
    y   = e:1/N:1
    par = params(d)
    o = map(y) do y
        quantile(d,y)
    end
    if permute
        permute!(o, randperm(N))
    end
    o
end
