using Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer,Distributions,Random,LinearAlgebra,OhMyREPL, ReactiveMP
enable_autocomplete_brackets(false);colorscheme!("GruvboxDark");

include("DiscreteLAIF.jl")
include("helpers.jl")

# Rule is missing from RxInfer
@rule Transition(:in, Marginalisation) (q_out::Any, q_a::Any) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end


@model function t_maze(θ_A,B,C,D,T)

    z_0 ~ Categorical(D)
    # We use datavar here since x~ Pointmass is not a thing
    x = datavar(Vector{Float64},T)
    z = randomvar(T)

    z_prev = z_0
    A ~ MatrixDirichlet(θ_A)

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        # We use the pipeline to only initialise the messages we need
        x[t] ~ DiscreteLAIF(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,3), (nothing, vague(Categorical,8), nothing)) }
        z_prev = z[t]
    end
    return z, z_0
end;



# Node constraints
@meta function t_maze_meta()
    DiscreteLAIF(x,z) -> PSubstitutionMeta()
end


T = 2
A,B,C,D = constructABCD(0.90,ones(T)*2,T);

initmarginals = (
                 z = [Categorical(fill(1/8,8)) for t in 1:T],
                 z_0 = [Categorical(fill(1/8,8))],
                 A = MatrixDirichlet(ones(size(A))),
                );

#initmessages = (
#                 z = [Categorical(fill(1/8,8)) for t in 1:T],
#                 z_0 = [Categorical(fill(1/8,8))],
#                );
#model = t_maze(A,Bs,C,D,T)
using Random
Random.seed!(123)
include("DiscreteLAIF.jl")
its=10
i = j = 2
Bs = (B[i],B[j])
result = inference(model = t_maze(A,Bs,C,D,T),
                   data= (x=C,),
                   initmarginals = initmarginals,
#                   initmessages = initmessages,
                   meta = t_maze_meta(),
                   free_energy=true,
                   iterations = its)

F = zeros(4,4);

for i in 1:4
    for j in 1:4
        Bs = (B[i],B[j])
        result = inference(model = t_maze(A,Bs,C,D,T),
                           data= (x=C,),
                           initmarginals = initmarginals,
                           meta = t_maze_meta(),
                           free_energy=true,
                           iterations = its)
        F[i,j] = result.free_energy[end] ./ log(2)
    end
end
F
