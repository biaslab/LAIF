using Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer,Distributions,Random,LinearAlgebra,OhMyREPL, ReactiveMP
enable_autocomplete_brackets(false);colorscheme!("GruvboxDark");

# include("GFECategorical.jl")
include("GFECategorical2.jl")
include("helpers.jl")

# gfepipeline = ReactiveMP.RequireEverythingFunctionalDependencies()
gfepipeline = GFEPipeline((2,))

A,B,C,D = constructABCD(1.00,[2.,2.],2)

@model function t_maze(A,B,C,T, pipeline = nothing)

    z_0 = datavar(Vector{Float64})
    # We use datavar here since x~ Pointmass is not a thing
    x = datavar(Vector{Float64},T)
    z = randomvar(T)
    w = randomvar(T)

    # Requires changes in the ReactiveMP core, `@meta` does not support pipelines (yet)
    pipeline = something(pipeline, ReactiveMP.DefaultFunctionalDependencies())

    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        w[t] ~ Transition(z[t], A) where { pipeline = pipeline }
        w[t] ~ Categorical(x[t])
        z_prev = z[t]
    end
end;

@constraints [ warn = false ] function t_maze_constraints()
    q(x, z, w, A) = q(x)q(w)q(z)q(A)
    q(w) :: EpistemicProduct
end

@meta function t_maze_meta()
    Transition(z, w) -> EpistemicMeta() 
    Categorical(x, w) -> EpistemicMeta()
end

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
        result = inference(model = t_maze(A,Bs,C,T, gfepipeline),
                           data= (z_0 = D, x=C),
                           initmarginals = initmarginals,
                           initmessages = initmessages,
                           constraints = t_maze_constraints(),
                           meta = t_maze_meta(),
                           free_energy=true,
                           iterations = its)
        F[i,j] = result.free_energy[end]
    end
end

F

