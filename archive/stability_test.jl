using Pkg;Pkg.activate("."); Pkg.instantiate()
using OhMyREPL, ReactiveMP, LinearAlgebra
using ForwardDiff
enable_autocomplete_brackets(false)

include("helpers.jl")
T = 2
A,B,C,D = constructABCD(0.9,[2.0,2.0],T);
jac = ForwardDiff.jacobian

#safelog(x) = log(x + exp(-16))
#
#σ(x) = exp.(x) / sum(exp.(x))
#
#σ(D) .* (1 .- σ(D))
#A
#D = σ(rand(4))
#function the_jac(z)
#    n = size(D)[1]
#
#    ρ_mat = σ(D) * σ(D)'
#    ρ_mat[diagind(ρ_mat)] = σ(D) .* (1 .- σ(D))
#
#    A
#
#    A'*A*D
#
#    jacobian
#
#    ForwardDiff.jacobian((x) -> log.(A*x), D)
#    ==
#    1 ./ A
#    (D .+ exp(-16))
#
#


A = [0.98 0.02;
     0.02 0.98]

C = [0.5, 0.5]

D = [0.1, 0.9]

safelog(x) = log(x + tiny)#exp(-16))

function gfe_marginal(A,C,D,q_in)

    x = A * q_in
    ρ = exp.(diag(A' * safelog.(A)) + A' *(safelog.(C) .- safelog.(x)))

    μ_out = clamp.(exp.(ρ),tiny,huge) / clamp.(sum(exp.(ρ)),tiny,huge)

    mvec = D .* μ_out
    q_in = mvec ./ sum(mvec)
end

n = 2
q_in = ones(n) ./ n

q_in = gfe_marginal(A,C[1],B[2]*D, q_in);
J = jac((x) -> gfe_marginal(A,C[1],B[2]*D, x), q_in);
eigvals(J)

q_in = gfe_marginal(A,C,D, q_in)
J = jac((x) -> gfe_marginal(A,C,D, x), q_in)
eigvals(J)
