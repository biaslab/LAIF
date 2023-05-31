using Pkg;Pkg.activate("..");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
import StatsBase: entropy
enable_autocomplete_brackets(false)

include("forward/forward_transition.jl")
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

# THERE HAS TO BE A BETTER WAY!!!
# Remove energy contributions from transition nodes
@average_energy Transition (q_out::Any, q_in::Any, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return 0.
end

@average_energy Transition (q_out_in::Contingency, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return 0.
end

@average_energy Transition (q_out_in::Contingency, q_a::PointMass, meta::ForwardOnlyMeta) = begin
    return 0.
end

@average_energy Transition (q_out::Any, q_in::Any, q_a::PointMass, meta::ForwardOnlyMeta) = begin
    return 0.
end

# Block edge entropies
entropy(d::DiscreteNonParametric) = 0.


#
#@model function t_maze(A,B,T)
#    Ac = constvar(A)
#
#    d = datavar(Vector{Float64})
#
#    x = datavar(Vector{Float64}, T)
#
#    z_0 ~ Categorical(d)
#    z = randomvar(T)
#
#    x = randomvar(T)
#    z_prev = z_0
#    for t in 1:T
#        z[t] ~ Transition(z_prev,B[t]) where{meta=ForwardOnlyMeta()}
#        x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireEverythingFunctionalDependencies()}
#        z_prev = z[t]
#    end
#end

@model function t_maze(A,D,B,T)

    z_0 ~ Categorical(D)

    z = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t]) where{meta=ForwardOnlyMeta()}
        x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireEverythingFunctionalDependencies(),meta=ForwardOnlyMeta()}
        z_prev = z[t]
    end
end;


initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
                );

initmessages = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
             );


imodel = Model(t_maze,A,D,[B[i],B[j]],T)

result = inference(model = imodel, data= (x = C,), initmarginals = initmarginals, initmessages = initmessages,free_energy=true, iterations = its)


# Try with all policies and evaluate EFE for each.
function evaluate_policies(B,its)
    F = zeros(4,4)
    for i in 1:4
        for j in 1:4
            imodel = Model(t_maze,A,D,[B[i],B[j]],T)

            result = inference(model = imodel, data= (x = C,), initmarginals = initmarginals, initmessages = initmessages,free_energy=true, iterations = its)

            F[i,j] =result.free_energy[end] ./log(2)
        end
    end
F
end

Fmap = evaluate_policies(B,20)
argmin(Fmap)


