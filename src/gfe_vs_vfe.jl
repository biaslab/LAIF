using Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer,Distributions,Random,LinearAlgebra,OhMyREPL, ReactiveMP
enable_autocomplete_brackets(false);colorscheme!("GruvboxDark");

include("GFECategorical.jl")
include("helpers.jl")

A,B,C,D = constructABCD(1.00,[2.,2.],2)

@model function t_maze(A,B,C,T)

    z_0 = datavar(Vector{Float64})
    # We use datavar here since x~ Pointmass is not a thing
    x = datavar(Vector{Float64},T)
    z = randomvar(T)

    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        x[t] ~ GFECategorical(z[t], A) where {pipeline=ReactiveMP.RequireEverythingFunctionalDependencies(), meta=ForwardOnlyMeta,q=MeanField()}
        z_prev = z[t]
    end
end;

T = 2

initmarginals = (
                 z = [Categorical(fill(1/8,8)) for t in 1:T],
                );

initmessages = (
                 z = [Categorical(fill(1/8,8)) for t in 1:T],
             );


# TODO: Figure out why FE increases with number of iterations - but only for the best policy?
its = 5
F = zeros(4,4);
for i in 1:4
    for j in 1:4
        Bs = (B[i],B[j])
        result = inference(model = t_maze(A,Bs,C,T),
                           data= (z_0 = D, x=C),
                           initmarginals = initmarginals,
                           initmessages = initmessages,
                           free_energy=true,
                           iterations = its)
        F[i,j] = result.free_energy[end]
    end
end

F

