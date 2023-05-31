using Pkg;Pkg.activate("..");Pkg.instantiate();
using RxInfer,ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions;
using Random
Random.seed!(666)
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


safelog(x) = log(clamp(x,tiny,Inf))

# Logmean but with guard rails in place
ReactiveMP.mean(::typeof(safelog), dist::MatrixDirichlet) = digamma.(clamp.(dist.a,tiny,Inf)) .- digamma.(sum(clamp.(dist.a,tiny,Inf); dims = 1))


## Standard entropy except 0s are set to tiny
function ReactiveMP.entropy(dist::MatrixDirichlet)
    a = clamp.(dist.a, tiny,Inf) # <-- Change is here
    return mapreduce(+, eachcol(a)) do column
        scolumn = sum(column)
        -sum((column .- 1.0) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) + sum(loggamma.(column))
    end
end

# Overwrite energy to avoid 0s
@average_energy MatrixDirichlet (q_out::MatrixDirichlet, q_a::PointMass) = begin
    H = mapreduce(+, zip(eachcol(clamp.(mean(q_a),tiny,Inf)), eachcol(mean(safelog, q_out)))) do (q_a_column, logmean_q_out_column) # Call to and a clamp safelog here
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end


include("helpers.jl");
include("DiscreteLAIF.jl");

@model function t_maze(θ_A,D,B,T)

    z_0 ~ Categorical(D)

    z = randomvar(T)

    x = datavar(Vector{Float64}, T)
    z_prev = z_0
    A ~ MatrixDirichlet(θ_A)

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        x[t] ~ DiscreteLAIF(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,3),vague(Categorical,8))}
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
its = 10

A,B,C,D = constructABCD(0.9,[2.0 for t in 1:T],T);

initmarginals = (
                 z = [Categorical(fill(1. /8. ,8)) for t in 1:T],
#                 A = vague(MatrixDirichlet,(16,8)),
                 A = MatrixDirichlet(A + 1e-2*rand(size(A)...)),
                );

Bs = (B[4],B[2])
#TODO: FE calculation breaks on entropy of A
result = inference(model = t_maze(A,D,Bs,T),
                   data= (x = C,),
                   initmarginals = initmarginals,
                   meta= t_maze_meta(),
                   free_energy = true,
#                   addons = (AddonMemory(),),
#                   constraints=pointmass_q(),
                   iterations=its
                  )

result.free_energy

# BEHOLD!!!!
probvec.(result.posteriors[:switch][end][1])
probvec.(result.posteriors[:switch][end][2])

########################################################################
# Below here is a giant mess of ongoing hacking, enter at your own risk!
########################################################################

bob = mean(result.posteriors[:A][1])
bob = result.posteriors[:A][end]
probvec(result.posteriors[:z][end][1])
probvec(result.posteriors[:z][end][end])

using ForneyLab
import ForneyLab: unsafeLogMean, labsgamma, ProbabilityDistribution, MatrixVariate, unsafeMean

using Random
Random.seed!(123)
marg_out = ProbabilityDistribution(MatrixVariate, Dirichlet, a=A);
marg_a = ProbabilityDistribution(MatrixVariate,PointMass,m=A + 1e-2*rand(size(A)...));

eng = energy(marg_out,marg_a)


function energy(marg_out, marg_a)
    (dims(marg_out) == dims(marg_a)) || error("Distribution dimensions must agree")

    log_mean_marg_out = unsafeLogMean(marg_out)

    H = 0.0
    for k = 1:dims(marg_out)[2] # For all columns
        a_sum = sum(marg_a.params[:m][:,k])

        H += -labsgamma(a_sum) +
        sum(labsgamma.(marg_a.params[:m][:,k])) -
        sum( (marg_a.params[:m][:,k] .- 1.0).*log_mean_marg_out[:,k] )
    end

    return H
end

k = 2
q_a_column = mean(q_a)[:,k]
a_sum = sum(q_a_column)
logmean_q_out_column = mean(safelog,q_out)[:,k]

loggamma(sum(q_a_column))
sum(loggamma.(q_a_column))
sum((q_a_column .- 1.0) .* logmean_q_out_column)

q_out = ReactiveMP.MatrixDirichlet(A)
mean(safelog,q_out)
Random.seed!(123)
q_a = ReactiveMP.PointMass(A + 1e-2*rand(size(A)...))

more_eng = more_energy(q_out,q_a)
eng - more_eng

function energy(marg_out, marg_a)
    (dims(marg_out) == dims(marg_a)) || error("Distribution dimensions must agree")

    log_mean_marg_out = unsafeLogMean(marg_out)

    H = 0.0
    for k = 1:dims(marg_out)[2] # For all columns
        a_sum = sum(marg_a.params[:m][:,k])

        H += -labsgamma(a_sum) +
        sum(labsgamma.(marg_a.params[:m][:,k])) -
        sum( (marg_a.params[:m][:,k] .- 1.0).*log_mean_marg_out[:,k] )
    end

    return H
end
# Overwrite energy to avoid 0s
function more_energy(q_out::MatrixDirichlet, q_a::ReactiveMP.PointMass)
    H = mapreduce(+, zip(eachcol(mean(q_a)), eachcol(mean(safelog, q_out)))) do (q_a_column, logmean_q_out_column)
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end


