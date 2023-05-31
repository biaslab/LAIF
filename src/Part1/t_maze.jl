using Pkg;Pkg.activate("/home/mkoudahl/biaslab/repos/EpistemicMessagePassing");Pkg.instantiate();
using RxInfer,ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions, Random
Random.seed!(666)
enable_autocomplete_brackets(false),colorscheme!("GruvboxDark");

include("transition_mixture/transition_mixture.jl")
include("transition_mixture/marginals.jl")
include("transition_mixture/in.jl")
include("transition_mixture/out.jl")
include("transition_mixture/switch.jl")
include("goal_observation.jl")
include("helpers.jl")


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


@model function t_maze(A,D,B1,B2,B3,B4,T)

    z_0 ~ Categorical(D)

    z = randomvar(T)
    switch = randomvar(T)

    c = datavar(Vector{Float64}, T)
    z_prev = z_0

    for t in 1:T
        switch[t] ~ Categorical(fill(1. /4. ,4))
	z[t] ~ TransitionMixture(z_prev,switch[t], B1,B2,B3,B4)
        c[t] ~ GoalObservation(z[t], A) where {pipeline = GeneralizedPipeline(vague(Categorical, 8)) }
        z_prev = z[t]
    end
end;



@constraints function pointmass_q()
    q(switch) :: PointMass
end

# Node constraints
@meta function t_maze_meta()
    GoalObservation(c,z) -> GeneralizedMeta()
end

T =10000;
its = 20;
initmarginals = (
#                 z_0 = Categorical(fill(1. /8. ,8)),
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
                );

A,B,C,D = constructABCD(0.9,[2.0 for t in 1:T],T);

@btime result = inference(model = t_maze(A,D,B[1],B[2],B[3],B[4],T),
                   data= (c = C,),
                   initmarginals = initmarginals,
                   meta= t_maze_meta(),
                   free_energy = true,
                   iterations=its,
                   options=(limit_stack_depth=300,)
                  )


# BEHOLD!!!!
probvec.(result.posteriors[:switch][end][1])
probvec.(result.posteriors[:switch][end][2])

using Plots, UnicodePlots
unicodeplots()
plot(result.free_energy)

# Try without pointmass constraints, still works
result = inference(model = t_maze(A,D,B[1],B[2],B[3],B[4],T),
                   data= (c = C,),
                   initmarginals = initmarginals,
                   meta= t_maze_meta(),
                   constraints=pointmass_q(),
                   iterations=its,
                  )

probvec(result.posteriors[:switch][end][1])
probvec(result.posteriors[:switch][end][2])
