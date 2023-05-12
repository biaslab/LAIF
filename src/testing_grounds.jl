using Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer,Distributions,Random,LinearAlgebra,OhMyREPL, ReactiveMP
import Distributions.entropy
import SpecialFunctions: digamma, loggamma
enable_autocomplete_brackets(false);colorscheme!("GruvboxDark");

include("DiscreteLAIF.jl")
include("helpers.jl")

# We need to get rid of zero entries in MatrixDirichlets to not break FE calculation
function Distributions.entropy(dist::MatrixDirichlet)
    return mapreduce(+, eachcol(dist.a)) do column
        # This isn't pretty but it gets the job done
        col = clamp.(column, tiny, Inf)
        scol = sum(col)
        -sum((col .- 1.0) .* (digamma.(col) .- digamma.(scol))) - loggamma(scol) + sum(loggamma.(col))
    end
end


@average_energy MatrixDirichlet (q_out::MatrixDirichlet, q_a::PointMass) = begin
    # Hacky McHackface
    q_out = MatrixDirichlet(clamp.(mean(q_out), tiny, Inf))
    q_a = MatrixDirichlet(clamp.(mean(q_a), tiny, Inf))
    H = mapreduce(+, zip(eachcol(mean(q_a)), eachcol(mean(log, q_out)))) do (q_a_column, logmean_q_out_column)
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end


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
    A ~ MatrixDirichlet(θ_A)

    z_prev = z_0

    for t in 1:T
        z[t] ~ Transition(z_prev,B[t])
        # We use the pipeline to only initialise the messages we need
        x[t] ~ DiscreteLAIF(z[t], A) where {q = MeanField(), pipeline = GFEPipeline((2,3),(Nothing,
                                                                                           vague(Categorical,8),
                                                                                           MatrixDirichlet(θ_A)
                                                                                          ))}
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
                 #A = MatrixDirichlet(A),
                 A = vague(MatrixDirichlet,size(A))
                );


its=50
using Random
Random.seed!(123)

i = 4
j =  2
Bs = (B[i],B[j])
result = inference(model = t_maze(A,Bs,C,D,T),
               data= (x=C,),
               initmarginals = initmarginals,
               meta = t_maze_meta(),
#               constraints = nonzero(),
               free_energy=true,
               iterations = its)

Ahat = result.posteriors[:A][end]
mean(Ahat)



# Instead of adding jitters, we should go through the constraints specification
#
#@constraints function nonzero()
#    q(A):: NonZeroConstraint()
#end


#struct NonZeroConstraint <: AbstractFormConstraint end
#
#ReactiveMP.is_point_mass_form_constraint(::NonZeroConstraint) = false
#ReactiveMP.default_form_check_strategy(::NonZeroConstraint)   = FormConstraintCheckLast()
#ReactiveMP.default_prod_constraint(::NonZeroConstraint)       = ProdGeneric()
#
## Adds a bit of jiter to avoid zeros
#function ReactiveMP.constrain_form(::NonZeroConstraint, distribution)
#    m = mean(distribution)
#    return MatrixDirichlet(m .+ tiny)
#end
#
