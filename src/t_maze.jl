using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("transition_mixture.jl")
include("categorical.jl")
include("helpers.jl")

T = 2

A,B,C,D = constructABCD(0.9,2.0,T)

# Try with all policies and evaluate EFE for each.
# Try with the EFE evaluation from the paper
@model function t_maze(A,D,B1,B2,B3,B4,T)
    z_0 ~ Categorical(D)

    z = randomvar(T)
    switch = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
	switch[t] ~ Categorical(fill(1. /4. ,4))
	z[t] ~ TransitionMixture(z_prev,switch[t], B1,B2,B3,B4)
        x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireInbound(in = Categorical(fill(1. /8. ,8)))}
        z_prev = z[t]
    end
end

imodel = Model(t_maze,A,D,B[1],B[2],B[3],B[4],T)

result = inference(model = imodel, data= (x = C,))

# Ignores first step, goes to cue on second
probvec(result.posteriors[:switch][1][1])

#probvec(result.posteriors[:z][1][2])

