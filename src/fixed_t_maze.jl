using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
#include("transition_mixture.jl")
include("categorical.jl")
include("helpers.jl")

T = 2

A,B,C,D = constructABCD(0.9,2.0,T)


# This is not the right rule, check ForneyLab doc
@rule Transition(:in, Marginalisation) (q_out::DiscreteNonParametric, q_a::PointMass) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::PointMass) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

# Try with all policies and evaluate EFE for each.
# Try with the EFE evaluation from the paper
@model [default_factorisation=MeanField()] function t_maze(A,D,B,T)
    Ac = constvar(A)
    Bc = constvar(B)

    z_0 ~ Categorical(D)
    z = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,Bc)
        x[t] ~ GFECategorical(z[t], Ac) where {pipeline=RequireInbound(in = Categorical(fill(1. /8. ,8)))}
        z_prev = z[t]
    end
end

imarginals = (
              z = vague(Categorical,8),
             )

imodel = Model(t_maze,A,D,B[1],T)
#constraints=t_maze_constraints(),
result = inference(model = imodel, data= (x = C,),initmarginals=imarginals,  free_energy=true)

# Why is this Inf?????
result.free_energy
