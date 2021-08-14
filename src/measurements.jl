
for PT in (:Particles, :StaticParticles)
    @eval begin
        function $PT(N, m::Measurements.Measurement{T})::$PT{T,N} where T
            $PT(N, Normal(Measurements.value(m),Measurements.uncertainty(m)))
        end
    end
end

"""
Convert an uncertain number from Measurements.jl to the equivalent particle representation with the default number of particles.
"""
function Particles(m::Measurements.Measurement{T}) where T
    Particles(DEFAULT_NUM_PARTICLES, m)
end

"""
Convert an uncertain number from Measurements.jl to the equivalent particle representation with the default number of particles.
"""
function StaticParticles(m::Measurements.Measurement{T}) where T
    StaticParticles(DEFAULT_STATIC_NUM_PARTICLES, m)
end

"""
    Measurements.value(p::AbstractParticles) = mean(p)
"""
Measurements.value(p::AbstractParticles) = pmean(p)

"""
    Measurements.uncertainty(p::AbstractParticles) = std(p)
"""
Measurements.uncertainty(p::AbstractParticles) = pstd(p)
