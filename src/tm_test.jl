using Pkg;Pkg.activate("..");Pkg.instantiate();
using RxInfer,ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions;
enable_autocomplete_brackets(false),colorscheme!("GruvboxDark");


# Need to make pointmass constraints for discrete vars
import RxInfer.default_point_mass_form_constraint_optimizer
import RxInfer.PointMassFormConstraint

function default_point_mass_form_constraint_optimizer(
    ::Type{Univariate},
    ::Type{Discrete},
    constraint::PointMassFormConstraint, distribution
)

    out = zeros( length(probvec(distribution)))
    out[argmax(probvec(distribution))] = 1.

    PointMass(out)
end


#include("tm_back.jl");
include("transition_mixture/transition_mixture.jl");
include("transition_mixture/marginals.jl");
include("transition_mixture/in.jl");
include("transition_mixture/out.jl");
include("transition_mixture/switch.jl");
#include("DiscreteLAIF.jl");
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
        #x[t] ~ DiscreteLAIF(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,),vague(Categorical,8))}
        x[t] ~ Transition(z[t],A)
        z_prev = z[t]
    end
end;




#@constraints function pointmass_q()
#    q(switch) :: PointMass
#end

T =2;

A,B,C,D = constructABCD(0.9,[2.0 for t in 1:T],T);
D = [1,0,0,0]
B1 =zeros(4,4) ; B1[1,:] .= 1.
B2 =zeros(4,4) ; B2[2,:] .= 1.
B3 =zeros(4,4) ; B3[3,:] .= 1.
B4 =zeros(4,4) ; B4[4,:] .= 1.
B = [B1,B2,B3,B4]

C[1] = [0,1,0,0]
C[2] = [0,0,1,0]
A = diageye(4)


result = inference(model = t_maze(A,D,B[1],B[2],B[3],B[4],T),
                   data= (x = C,),
                   free_energy = true,
#                   constraints=pointmass_q(),
                   iterations=10,
                   )
# BEHOLD!!!!
probvec.(result.posteriors[:switch][end][1])
probvec.(result.posteriors[:switch][end][2])
