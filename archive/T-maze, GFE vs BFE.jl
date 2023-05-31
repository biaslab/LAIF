using Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer,Distributions,Random,LinearAlgebra, ReactiveMP, OhMyREPL
enable_autocomplete_brackets(false);colorscheme!("GruvboxDark")

include("GFECategorical.jl")
include("helpers.jl")
Random.seed!(666) ;


# Rule is missing from RxInfer
@rule Transition(:in, Marginalisation) (q_out::Any, q_a::Any) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::PointMass, ) = begin
    a = clamp.(exp.(mean(log, q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

# Model for the T-maze experiment
@model function t_maze(A,B,C,T, pipeline = nothing)

    D_0 = datavar(Vector{Float64})
    z_0 ~ Categorical(D_0)
    # We use datavar here since x~ Pointmass is not a thing
    x = datavar(Vector{Float64},T)
    z = randomvar(T)
    w = randomvar(T)

    # Requires changes in the ReactiveMP core, `@meta` does not support pipelines (yet)
    pipeline = something(pipeline, ReactiveMP.DefaultFunctionalDependencies())

    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev, B[t])
        w[t] ~ Transition(z[t], A) where { pipeline = pipeline }
        w[t] ~ Categorical(x[t])
        z_prev = z[t]
    end
end;

# Edge Constraints
@constraints [ warn = false ] function gfeconstraints()
    q(x, z, w, A) = q(x)q(w)q(z)q(A)
    q(w) :: PSubstitutionProduct
end

# Node constraints
@meta function gfemeta()
    Transition(z, w) -> PSubstitutionMeta()
    Categorical(x, w) -> PSubstitutionMeta()
end

@constraints [ warn = false ] function bfeconstraints()
    q(x, z, w, A) = q(x)q(w)q(z)q(A)
end

# Custom pipeline so we can request incoming message only on the edge we need
gfepipeline = GFEPipeline((2,));

# Get required matrices
A,B,C,D = constructABCD(0.9,[2.,2.],2);

# Number of inference iterations
its = 5

# Planning horizon
T = 2

# Choose functional
gfe_setup = (
    pipeline = gfepipeline,
    constraints = gfeconstraints(),
    meta = gfemeta()
);

bfe_setup = (
    pipeline = nothing,
    constraints = bfeconstraints(),
    meta = nothing
);


# Initialise marginals and messages
initmarginals = (
                 z = [Categorical(fill(1/8,8)) for t in 1:T],
                );

initmessages = (
                 z = [Categorical(fill(1/8,8)) for t in 1:T],
             );


# Select between GFE and BFE experiments
current_setup = gfe_setup;

F = zeros(4,4);
for i in 1:4
    for j in 1:4
        Bs = (B[i],B[j])
        global result = inference(model = t_maze(A,Bs,C,T, current_setup[:pipeline]),
                           data= (D_0 = D, x = C),
                           initmarginals = initmarginals,
                           initmessages = initmessages,
                           constraints = current_setup[:constraints],
                           meta = current_setup[:meta],
                           free_energy=true,
                           iterations = its)
        F[i,j] = result.free_energy[end] / log(2)
    end
end

# Print free energy of all policies
F
