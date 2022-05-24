using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("transition_mixture.jl")
include("categorical.jl")
include("helpers.jl")


A,B,C,D = constructABCD(1.0,2.)

@model function t_maze(A,D,B1,B2,B3,B4)
    z_0 ~ Categorical(D)

    z = randomvar(2)
    switch = randomvar(2)

    x = datavar(Vector{Float64}, 2)
    z_prev = z_0

    for t in 1:2
	switch[t] ~ Categorical(fill(1. /4. ,4))
	z[t] ~ TransitionMixture(z_prev,switch[t], B1,B2,B3,B4)
        x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireInbound(in = Categorical(fill(1. /8. ,8)))}
        z_prev = z[t]
    end
end

imodel = Model(t_maze,A,D,B[1],B[2],B[3],B[4])

result = inference(model = imodel, data= (x = C,))

probvec(result.posteriors[:switch][1][2])


