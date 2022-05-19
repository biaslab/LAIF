using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("transition_mixture.jl")

# Hack together some transition matrices
Bs = [zeros(4,4) for x in 1:4]
for i in 1:4
    Bs[i][i,:] .= 1.
end

# Initial state
D = [1,0,0,0]
# Likelihood
A = diageye(4)

goal = [[0.,0.,0.,1.]]

@model function controlled_hmm(A,D,B1,B2,B3,B4,n)

    z_0 ~ Categorical(D)
    switch ~ Categorical(fill(1. /4. ,4))

    z = randomvar(n)
    x = datavar(Vector{Float64}, n)

    z_prev = z_0

    for t in 1:n
	z[t] ~ TransitionMixture(z_prev,switch, B1,B2,B3,B4)
        x[t] ~ Transition(z[t], A)
        z_prev = z[t]
    end

end

imodel = Model(controlled_hmm,A,D,Bs[1],Bs[2],Bs[3],Bs[4],1)

result = inference(model = imodel, data= (x = goal,))

probvec(result.posteriors[:switch][1])
probvec(result.posteriors[:z][1][1])

