using ForwardDiff: jacobian
using TupleTools: deleteat
using ReactiveMP: AbstractNodeFunctionalDependenciesPipeline, RequireMarginalFunctionalDependencies, messagein, setmessage!, get_samples, get_weights
import ReactiveMP: message_dependencies, marginal_dependencies

include("distributions.jl")


struct GoalObservation end

@node GoalObservation Stochastic [c, z, A]


#----------
# Modifiers
#----------

# Metas
struct BetheMeta end # Forces explicit constraint specification to prevent mixups
struct ObservedMeta{I}
    x::I # Pointmass value for observation
end
struct GeneralizedMeta
    newton_iterations::Int64
end
GeneralizedMeta() = GeneralizedMeta(20) # Default number of iterations

# Pipelines
struct BethePipeline <: AbstractNodeFunctionalDependenciesPipeline end
struct GeneralizedPipeline <: AbstractNodeFunctionalDependenciesPipeline
    init_message::Categorical
end

function message_dependencies(::BethePipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return ()
end

# Bethe update rules for goal-observation node require marginals on all edges
function marginal_dependencies(::BethePipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return nodelocalmarginals
end

# Generalized update rule for state requires inbound message
function message_dependencies(pipeline::GeneralizedPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    if iindex === 2 # Message towards state
        input = ReactiveMP.messagein(nodeinterfaces[iindex])
        ReactiveMP.setmessage!(input, pipeline.init_message)
        return (nodeinterfaces[iindex],) # Include inbound message on state
    else
        return ()
    end
end

# Generalized update rule for state requires inbound marginal
function marginal_dependencies(::GeneralizedPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    if (iindex === 2) || (iindex === 3) # Message towards state or parameter
        return nodelocalmarginals # Include all marginals
    else
        return deleteat(nodelocalmarginals, cindex) # Include default marginals
    end
end


#-------------------
# Bethe Update Rules
#-------------------

@rule GoalObservation(:c, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Categorical, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    log_A = mean(log, q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return Dirichlet(x .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Categorical, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    log_A = mean(log, q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return Categorical(softmax(log_A'*x))
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Categorical, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::BetheMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    log_A = mean(log, q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return MatrixDirichlet(x*z' .+ 1)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Categorical, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::BetheMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    log_A = mean(log, q_A)

    # Compute internal marginal
    x = softmax(log_A*z + log_c)

    return -x'*(log_A*z + log_c - safelog.(x))
end


#----------------------
# Observed Update Rules
#----------------------

@rule GoalObservation(:c, Marginalisation) (q_z::Categorical, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::ObservedMeta) = begin
    return Dirichlet(meta.x .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::ObservedMeta) = begin
    log_A = mean(log, q_A)

    return Categorical(softmax(log_A'*meta.x))
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Categorical, 
                                            meta::ObservedMeta) = begin
    z = probvec(q_z)

    return MatrixDirichlet(meta.x*z' .+ 1)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Categorical, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::ObservedMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    log_A = mean(log, q_A)

    return -meta.x'*(log_A*z + log_c)
end


#-------------------------
# Generalized Update Rules
#-------------------------

@rule GoalObservation(:c, Marginalisation) (q_z::Categorical, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta) = begin
    z = probvec(q_z)
    A = mean(q_A)

    return Dirichlet(A*z .+ 1)
end

@rule GoalObservation(:z, Marginalisation) (m_z::Categorical,
                                            q_c::Union{Dirichlet, PointMass},
                                            q_z::Categorical,
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                            meta::GeneralizedMeta) = begin
    d = probvec(m_z)
    log_c = mean(log, q_c)
    z_0 = probvec(q_z)
    (A, h_A) = mean_h(q_A)

    # Root-finding problem for marginal statistics
    g(z) = z - softmax(-h_A + A'*log_c - A'*safelog.(A*z) + safelog.(d))

    z_k = deepcopy(z_0)
    for k=1:meta.newton_iterations
        z_k = z_k - inv(jacobian(g, z_k))*g(z_k) # Newton step for multivariate root finding
    end

    # Compute outbound message statistics
    rho = softmax(safelog.(z_k) - log.(d .+ 1e-6))

    return Categorical(rho)
end

@rule GoalObservation(:A, Marginalisation) (q_c::Union{Dirichlet, PointMass},
                                            q_z::Categorical, 
                                            q_A::Union{SampleList, MatrixDirichlet, PointMass},
                                            meta::GeneralizedMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    A_bar = mean(q_A)                                            

    log_mu(A) = (A*z)'*(log_c - safelog.(A_bar*z)) - z'*h(A)

    return ContinuousMatrixvariateLogPdf(log_mu)
end

@average_energy GoalObservation (q_c::Union{Dirichlet, PointMass}, 
                                 q_z::Categorical, 
                                 q_A::Union{SampleList, MatrixDirichlet, PointMass}, 
                                 meta::GeneralizedMeta) = begin
    log_c = mean(log, q_c)
    z = probvec(q_z)
    (A, h_A) = mean_h(q_A)

    return z'*h_A - (A*z)'*(log_c - safelog.(A*z))
end