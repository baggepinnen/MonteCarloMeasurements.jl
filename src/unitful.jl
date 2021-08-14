import .Unitful: Quantity, FreeUnits

function to_num_str(p::AbstractParticles{T}, d=3, ds=d-1) where T <: Quantity
    s = pstd(p)
    if s.val < eps(p)
        string(pmean(p))
    else
        string(pmean(p), " ± ", s)
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

        function Unitful.uconvert(a::Unitful.FreeUnits, y::$PT)
            $PT(Unitful.uconvert.(a, y.particles))
        end
    end

    for op in (*, /)
        f = nameof(op)
        @eval begin
            function Base.$f(p::$PT{T,N}, y::Quantity{S,D,U}) where {S, D, U, T, N}
                NT = promote_type(T, S)
                $PT{Quantity{NT,D,U},N}($(op).(p.particles , y))
            end

            function Base.$f(p::$PT{T,N}, y::Quantity{S,D,U}) where {S, D, U, T <: Quantity, N}
                QT = Base.promote_op($op, T, typeof(y))
                $PT{QT,N}($(op).(p.particles, y))
            end

            # Below is just the reverse signature of above
            function Base.$f(y::Quantity{S,D,U}, p::$PT{T,N}) where {S, D, U, T <: Quantity, N}
                QT = Base.promote_op($op, typeof(y), T)
                $PT{QT,N}($(op).(y, p.particles))
            end

            function Base.$f(p::$PT, y::FreeUnits)
                $PT($(op).(p.particles, y))
            end
        end

    end
end
