using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("categorical.jl")
include("helpers.jl")

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


@model function t_maze(A,B,D,T)
    Ac = constvar(A)

    z_0 ~ Categorical(D)
    z = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireEverythingFunctionalDependencies()}
        z_prev = z[t]
    end
end

initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T]
                 ,
                );

initmessages = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T]
                 ,
             );

# Try with all policies and evaluate GBFE for each.
function evaluate_policies(B,its)
    F = zeros(4,4)
    for i in 1:4
        for j in 1:4
            imodel = Model(t_maze,A,[B[i],B[j]],D,T)

            result = inference(model = imodel, data= (x = C,), initmarginals = initmarginals, initmessages = initmessages, free_energy=true, iterations=its)

            F[i,j] =result.free_energy[end] ./log(2)
        end
    end
F
end

evaluate_policies(B,4)
argmin(evaluate_policies(B,1))
