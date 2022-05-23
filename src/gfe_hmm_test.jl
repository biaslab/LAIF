using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("transition_mixture.jl")
include("categorical.jl")

# Hack together some transition matrices
Bs = [zeros(4,4) for x in 1:4]
for i in 1:4
    Bs[i][i,:] .= 1.
end

# Initial state
D = [1.,0,0,0]
# Likelihood
A = diageye(4)

goal = [[0.,0.,0.5,0.5],
	[1.,0.,0.0,0.0],
	[0.,0.,0.3,0.7] ]

#struct GFECategorical end
#@node GFECategorical Stochastic [out,in,A]
#
#safelog(x) = log(x +eps())
#@rule GFECategorical(:in, Marginalisation) (q_out::PointMass,m_in::Categorical, q_A::PointMass) = begin
#    z = probvec(m_in)
#    A = mean(q_A)
#    C = probvec(q_out)
#    # q_out needs to be A*mean(incoming), hence this line
#    x = A * z
#    # Write this out in a nicer way. Vec is there to force the type to not be Matrix
#    ρ = vec(sum(z .* A .* safelog.(A)',dims= 2)) + z.*A * (safelog.(C) - safelog.(x))
#    return Categorical(exp.(ρ) / sum(exp.(ρ)))
#end


@model function controlled_hmm(A,D,B1,B2,B3,B4,n)

    z_0 ~ Categorical(D)

    z = randomvar(n)
    switch = randomvar(n)

    x = datavar(Vector{Float64}, n)
    z_prev = z_0

    for t in 1:n
	switch[t] ~ Categorical(fill(1. /4. ,4))
	z[t] ~ TransitionMixture(z_prev,switch[t], B1,B2,B3,B4)
	x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireInbound(in = Categorical(fill(1. /4. ,4)))}
        z_prev = z[t]
    end
end

imodel = Model(controlled_hmm,A,D,Bs[1],Bs[2],Bs[3],Bs[4],3)

result = inference(model = imodel, data= (x = goal,))

probvec(result.posteriors[:switch][1][3])


