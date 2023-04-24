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
# Any 0s in A break FE computation. Therefore we need to clamp all 0's to tinys instead

# Functions we need to modify
import ReactiveMP: entropy, mean
import SpecialFunctions: loggamma, logabsgamma, digamma

ReactiveMP.mean(::typeof(safelog), dist::MatrixDirichlet) = digamma.(clamp.(dist.a,tiny,Inf)) .- digamma.(sum(clamp.(dist.a,tiny,Inf); dims = 1))

# Standard entropy except 0s are set to tiny
function ReactiveMP.entropy(dist::MatrixDirichlet)
    a = clamp.(dist.a, tiny,Inf)
    return mapreduce(+, eachcol(a)) do column
        scolumn = sum(column)
        -sum((column .- 1.0) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) + sum(loggamma.(column))
    end
end

# Overwrite energy to avoid 0s
@average_energy MatrixDirichlet (q_out::MatrixDirichlet, q_a::PointMass) = begin
    H = mapreduce(+, zip(eachcol(clamp.(mean(q_a),tiny,Inf)), eachcol(mean(safelog, q_out)))) do (q_a_column, logmean_q_out_column)
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end


include("transition_mixture.jl");
include("helpers.jl");
include("DiscreteLAIF.jl");

@model function t_maze(θ_A,D,B,T)

    z_0 ~ Categorical(D)

    z = randomvar(T)
    switch = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0
    A ~ MatrixDirichlet(θ_A)

    for t in 1:T
        switch[t] ~ Categorical(fill(1. /4. ,4))
        z[t] ~ TransitionMixture(z_prev,switch[t], B[1],B[2],B[3],B[4])
        x[t] ~ DiscreteLAIF(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,3),(Nothing,vague(Categorical,8),vague(MatrixDirichlet,size(θ_A))))
                                                                                   }
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

T = 2
its = 5

A,B,C,D = constructABCD(0.9,[2.0 for t in 1:T],T);

initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
                 #A = vague(MatrixDirichlet,(16,8)),
                 A = MatrixDirichlet(A + 1e-2*rand(size(A)...)),
                );

#TODO: FE calculation breaks on entropy of A
result = inference(model = t_maze(A,D,B,T),
                   data= (x = C,),
                   initmarginals = initmarginals,
                   meta= t_maze_meta(),
                   free_energy = true,
#                   addons = (AddonMemory(),),
#                   constraints=pointmass_q(),
                   iterations=its
                  )


# BEHOLD!!!!
probvec.(result.posteriors[:switch][end][1])
probvec.(result.posteriors[:switch][end][2])
bob = mean(result.posteriors[:A][end])
bob = result.posteriors[:A][end]
