using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("transition_mixture.jl")
include("approx_marginal_categorical.jl")
include("helpers.jl")

T = 2

A,B,C,D = constructABCD(0.9,2.0,T)

# Try with all policies and evaluate EFE for each.
# Try with the EFE evaluation from the paper
@model function [default_factorisation=MeanField()] t_maze(Ac,D,T)
    A = constvar(Ac)
    z_0 ~ Categorical(D)

    z = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    B ~ MatrixDirichlet(diageye(8))
    for t in 1:T
	z[t] ~ Transition(z_prev,B)
        x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireInbound(in = Categorical(fill(1. /8. ,8))), meta=GFEMeta(Categorical(fill(1. /8. ,8)))}
        z_prev = z[t]
    end
end

imodel = Model(t_maze,A,D,T)
imarginals = (
    B = vague(MatrixDirichlet, 8, 8),
)


result = inference(model = imodel, data= (x = C,), initmarginals=imarginals)

# Ignores first step, goes to cue on second
#probvec(result.posteriors[:switch][1][2])
#probvec(result.posteriors[:z][1][1])

