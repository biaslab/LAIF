using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("categorical.jl")
include("helpers.jl")

T = 2;

A,B,C,D = constructABCD(0.9,[2.0,2.0],T);

#A = Matrix{Float64}(vcat(I(8),I(8)))

# Workaround for ReactiveMP being unable to compute FE when D contains 0 entries
DD = ones(8) * eps();
DD[1:2] .= 0.5 - eps() * 6;
DD;

# Variatonal update rules for messing with VMP
@rule Transition(:in, Marginalisation) (q_out::DiscreteNonParametric, q_a::PointMass) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::PointMass) = begin
    a = clamp.(exp.(mean(log, q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

# Try with all policies and evaluate EFE for each.
# Try with the EFE evaluation from the paper
@model function t_maze(A,D,B,T)
    Ac = constvar(A)

    z_0 ~ Categorical(D)
    z = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        x[t] ~ GFECategorical(z[t], Ac) where {pipeline=RequireMarginal(in = Categorical(fill(1. /8. ,8)))}
        z_prev = z[t]
    end
end

#@constraints function efe_constraints()
#    q(z_0, z) = q(z_0)q(z)
#end


function evaluate_policies(B,its)
    F = zeros(4,4)
    for i in 1:4
        for j in 1:4
            imodel = Model(t_maze,A,D,[B[i],B[j]],T)
            result = inference(model = imodel, data= (x = C,),free_energy=true,iterations=its)

            F[i,j] =result.free_energy[end] ./log(2)
        end
    end
F
end

evaluate_policies(B,10)
argmin(evaluate_policies(B,10))
