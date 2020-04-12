import .Unitful: Quantity, FreeUnits

function to_num_str(p::AbstractParticles{T}, d=3) where T <: Quantity
    s = std(p)
    if T <: AbstractFloat && s < eps(p)
        string(mean(p))
    else
        string(mean(p), " ± ", s)
    end
end

for PT in ParticleSymbols

    @eval begin
        function Base.promote_rule(::Type{Quantity{S,D,U}}, ::Type{$PT{T,N}}) where {S, D, U, T, N}
            NT = promote_type(Quantity{S,D,U},T)
            $PT{NT,N}
        end

        function Base.convert(::Type{$PT{Quantity{S,D,U},N}}, y::Quantity) where {S, D, U, T, N}

            $PT{Quantity{S,D,U},N}(fill(y, N))

        end

        function Base.:*(p::$PT{T,N}, y::Quantity{S,D,U}) where {S, D, U, T, N}
            NT = promote_type(S,T)
            $PT{Quantity{NT,D,U},N}(p.particles .* y)
        end

        function Base.:*(p::$PT, y::FreeUnits)
            $PT(p.particles .* y)
        end

    end
end
