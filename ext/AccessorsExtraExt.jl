module AccessorsExtraExt

using MonteCarloMeasurements
using AccessorsExtra

function MonteCarloMeasurements.bymap₊(f, args...)
	Ns = @getall args |> RecursiveOfType(AbstractParticles) |> length(_.particles)
    allequal(Ns) || throw(ArgumentError("different number of particles within function arguments: $Ns"))
	N = first(Ns)
	vals = map(1:N) do i
		curargs = @modify(args |> RecursiveOfType(AbstractParticles)) do p
			p.particles[i]
		end
		f(curargs...)
	end
	v = first(vals)
	numoptics = AccessorsExtra.flat_concatoptic(v, RecursiveOfType(Number))
	valps = map(o -> Particles(o.(vals)), AccessorsExtra._optics(numoptics))
	return setall(v, numoptics, valps)
end

end
