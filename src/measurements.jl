
for PT in (:Particles, :StaticParticles)
    @eval begin
        function $PT(N, m::Measurements.Measurement{T})::$PT{T,N} where T
            $PT(N, Normal(Measurements.value(m),Measurements.uncertainty(m)))
        end
    end
end

function Particles(m::Measurements.Measurement{T}) where T
    Particles(DEFAULT_NUM_PARTICLES, m)
end

function StaticParticles(m::Measurements.Measurement{T}) where T
    StaticParticles(DEFAULT_STATIC_NUM_PARTICLES, m)
end

Measurements.value(p::AbstractParticles) = mean(p)
Measurements.uncertainty(p::AbstractParticles) = std(p)
