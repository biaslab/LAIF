using Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer,Distributions,Random,LinearAlgebra,OhMyREPL, ReactiveMP
enable_autocomplete_brackets(false);colorscheme!("GruvboxDark");

include("GFEComposite.jl")
include("helpers.jl")

# Rule is missing from RxInfer
@rule Transition(:in, Marginalisation) (q_out::Any, q_a::Any) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

A,B,C,D = constructABCD(0.90,[2.,2.],2)

@model function t_maze(A,B,C,D,T)

    z_0 ~ Categorical(D)
    # We use datavar here since x~ Pointmass is not a thing
    x = datavar(Vector{Float64},T)
    z = randomvar(T)

    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])# where {q = MeanField()}
        x[t] ~ Observation(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,),vague(Categorical,8))}
        z_prev = z[t]
    end
    return z, z_0
end;

# Node constraints
@meta function t_maze_meta()
    Observation(x,z) -> PSubstitutionMeta()
end


T = 2

initmarginals = (
                 z = [Categorical(fill(1/8,8)) for t in 1:T],
                 z_0 = [Categorical(fill(1/8,8))],
                );

i = 4
j= 2
its=10
Bs = (B[i],B[j])

result = inference(model = t_maze(A,Bs,C,D,T),
                   data= (x=C,),
                   initmarginals = initmarginals,
                   meta = t_maze_meta(),
                   free_energy=true,
                   iterations = its)

using Plots, UnicodePlots
unicodeplots()
plot(result.free_energy ./ log(2))
