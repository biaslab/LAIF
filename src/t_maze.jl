using Pkg;Pkg.activate("..");Pkg.instantiate();
using RxInfer,ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions;
enable_autocomplete_brackets(false),colorscheme!("GruvboxDark");


# Need to make pointmass constraints for discrete vars
import RxInfer.default_point_mass_form_constraint_optimizer
import RxInfer.PointMassFormConstraint

function default_point_mass_form_constraint_optimizer(
    ::Type{Univariate},
    ::Type{Discrete},
    constraint::PointMassFormConstraint,
    distribution
)

    out = zeros( length(probvec(distribution)))
    out[argmax(probvec(distribution))] = 1.

    PointMass(out)
end


include("transition_mixture.jl");
include("DiscreteLAIF.jl");
include("helpers.jl");

@model function t_maze(A,D,B1,B2,B3,B4,T)

    z_0 ~ Categorical(D)

    z = randomvar(T)
    switch = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
        switch[t] ~ Categorical(fill(1. /4. ,4))
	z[t] ~ TransitionMixture(z_prev,switch[t], B1,B2,B3,B4)
        x[t] ~ DiscreteLAIF(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,),vague(Categorical,8))}
        z_prev = z[t]
    end
end;

initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
                );



@constraints function pointmass_q()
    q(switch) :: PointMass
end

# Node constraints
@meta function t_maze_meta()
    DiscreteLAIF(x,z) -> PSubstitutionMeta()
end

T = 100_000;

A,B,C,D = constructABCD(0.9,[2.0 for t in 1:T],T);


result = inference(model = t_maze(A,D,B[1],B[2],B[3],B[4],T),
                   data= (x = C,),
                   initmarginals = initmarginals,
                   meta= t_maze_meta(),
#                   constraints=pointmass_q(),
                   iterations=5,
                   options=(limit_stack_depth=500,))


# BEHOLD!!!!
probvec.(result.posteriors[:switch][end][1])
probvec.(result.posteriors[:switch][end][2])


# Try without pointmass constraints, still works
#result = inference(model = imodel, data= (x = C,), initmarginals = initmarginals, initmessages = initmessages, iterations=2)
#
#probvec(result.posteriors[:switch][end][1])
#probvec(result.posteriors[:switch][end][2])
