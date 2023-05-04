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


include("tm_back.jl");
include("transition_mixture/marginals.jl");
include("transition_mixture/transition_mixture.jl");
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




@constraints function pointmass_q()
    q(switch) :: PointMass
end

# Node constraints
@meta function t_maze_meta()
    DiscreteLAIF(x,z) -> PSubstitutionMeta()
end

T =2;# 100_000;
initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
                );

A,B,C,D = constructABCD(0.9,[2.0 for t in 1:T],T);

@marginalrule TransitionMixture(:out_in_z) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin

    I_ = length(probvec(m_out))
    J  = length(probvec(m_in))
    K  = length(probvec(m_z))
    μ_out = probvec(m_out)
    μ_in = probvec(m_in)
    μ_z = probvec(m_z)

    A_tilde = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]
    B = zeros(I_,J,K)
    for i in 1:I_
        for j in 1:J
            for k in 1:K
                B[i,j,k] = μ_out[i]*μ_in[j]*μ_z[k] * A_tilde[i,j,k]
            end
        end
    end
    return ContingencyTensor(B ./ sum(B))
end

@average_energy TransitionMixture (q_out_in_z::ContingencyTensor, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass) = begin

    # Make it work, make it right, make it fast. First we make it work, then we get rid of all the hacks
    log_A_bar = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]

    B = mean(q_out_in_z)
    U = 0
    for i in 1:4
       U+= -tr(B[:,:,i]' *log_A_bar[:,:,i] )
    end
    return U
end

result = inference(model = t_maze(A,D,B[1],B[2],B[3],B[4],T),
                   data= (x = C,),
                   initmarginals = initmarginals,
                   meta= t_maze_meta(),
                   free_energy = true,
#                   constraints=pointmass_q(),
                   iterations=50,
                   #options=(limit_stack_depth=500,)
                   )

# BEHOLD!!!!
probvec.(result.posteriors[:switch][end][1])
probvec.(result.posteriors[:switch][end][2])


# Try without pointmass constraints, still works
#result = inference(model = imodel, data= (x = C,), initmarginals = initmarginals, initmessages = initmessages, iterations=2)
#
#probvec(result.posteriors[:switch][end][1])
#probvec(result.posteriors[:switch][end][2])
