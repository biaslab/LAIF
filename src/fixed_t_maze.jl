using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)

include("GFECategorical.jl")
include("helpers.jl")

T = 2;

A,B,C,D = constructABCD(0.9,[2.0,2.0],T);

# Variatonal update rules for messing with VMP
@rule Transition(:in, Marginalisation) (q_out::DiscreteNonParametric, q_a::PointMass) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::PointMass) = begin
    a = clamp.(exp.(mean(log, q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end


# TODO rewrite this for arbitrary T
@model function t_maze(A,B,C,T)
    Ac = constvar(A)

    d = datavar(Vector{Float64})

    z_0 ~ Categorical(d)
    z = randomvar(T)

    #x = datavar(Vector{Float64}, T)
    x_1 = constvar(C[1])
    x_2 = constvar(C[2])

    #x = randomvar(T)
    z_prev = z_0

    z[1] ~ Transition(z_0,B[1])
    x_1 ~ GFECategorical(z[1], A) where {pipeline=RequireEverythingFunctionalDependencies()}

    z[2] ~ Transition(z[1],B[2])
    x_2 ~ GFECategorical(z[2], A) where {pipeline=RequireEverythingFunctionalDependencies()}
    #for t in 1:T
    #    z[t] ~ Transition(z_prev,B[t])
    #    x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireEverythingFunctionalDependencies()}
    #    z_prev = z[t]
    #end
end

initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
                );

initmessages = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
             );

# Try with all policies and evaluate EFE for each.
function evaluate_policies(B,its)
    F = zeros(4,4)
    for i in 1:4
        for j in 1:4
            imodel = Model(t_maze,A,[B[i],B[j]],C,T)

            result = inference(model = imodel, data= (d = D,), initmarginals = initmarginals, initmessages = initmessages,free_energy=true, iterations = its)

            F[i,j] =result.free_energy[end] ./log(2)
        end
    end
F
end

Fmap = evaluate_policies(B,20)
argmin(Fmap)

