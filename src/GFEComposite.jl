using ForwardDiff: jacobian
using DomainSets: FullSpace
include("matrixlogpdf.jl")

#TODO: Make delta constrained updates/energy and BFE updates/energy

# Use custom pipeline to get marginal and message on incoming edge
struct GFEPipeline{I,M} <: ReactiveMP.AbstractNodeFunctionalDependenciesPipeline
    indices::I
    init_message::M
end

function ReactiveMP.message_dependencies(pipeline::GFEPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # We simply override the messages dependencies with the provided indices
    for index in pipeline.indices
        output = ReactiveMP.messagein(nodeinterfaces[index])
        ReactiveMP.setmessage!(output, pipeline.init_message)
    end
    return map(inds -> map(i -> @inbounds(nodeinterfaces[i]), inds), pipeline.indices)
end

function ReactiveMP.marginal_dependencies(pipeline::GFEPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # For marginals we require marginal on the same edges as in the provided indices
    require_marginal = ReactiveMP.RequireMarginalFunctionalDependencies(pipeline.indices, map(_ -> nothing, pipeline.indices))
    return ReactiveMP.marginal_dependencies(require_marginal, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
end
# Helper functions
# We don't want log(0) to happen
safelog(x) = log(clamp(x,tiny,Inf))
softmax(x) = exp.(x) ./ sum(exp.(x))

# Metas to change behaviour between BFE and GFE
struct PSubstitutionMeta end

# Optionally add data constraints
struct BetheMeta
    δ
end

# Constructor for default case, no data constraints
# TODO: make sure BetheMeta does not take any arbitrary thing
function BetheMeta()
    return BetheMeta(Missing)
end

struct Observation end
# Define node
@node Observation Stochastic [out,in, A]

### GFE SECTION
# GFE Energy
@average_energy Observation (q_out::Union{Dirichlet,PointMass}, q_in::Categorical, q_A::Union{PointMass,MatrixDirichlet},meta::PSubstitutionMeta) = begin
    s = probvec(q_in)
    A = mean(q_A)
    c = mean(q_out)

    -s' * diag(A' * safelog.(A)) + (A*s)'*(safelog.(A*s) .- safelog.(c))
end


# GFE Message towards A
@rule Observation(:A, Marginalisation) (q_out::Union{PointMass,Dirichlet}, m_in::Categorical, q_in::Categorical, q_A::MatrixDirichlet, meta::PSubstitutionMeta) = begin
    A_bar = mean(q_A)
    c = mean(q_out)
    s = probvec(q_in)

    # LogPdf
    logpdf(A) = s' *( diag(A'*safelog.(A)) + A'*(safelog.(c) - safelog.(A_bar*s)))
    return ContinuousMatrixvariateLogPdf(FullSpace(),logpdf)
end

# GFE Message towards input
@rule Observation(:in, Marginalisation) (q_out::Union{Dirichlet,PointMass},m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::PSubstitutionMeta) = begin

    d = probvec(m_in)
    s = probvec(q_in)
    A = mean(q_A)
    C = mean(q_out)

    # Newton iterations for stability
    g(s) = s - softmax(diag(A'*safelog.(A)) + A'*safelog.(C) - A'*safelog.(A*s) + safelog.(d))

    s_k = deepcopy(s)
    for i in 1:20 # TODO make this user specified
        s_k = s_k - inv(jacobian(g,s_k)) * g(s_k)
    end

    ρ = s_k ./ (d .+ 1e-6)

    return Categorical(ρ ./ sum(ρ))
end


# Message towards output (goal parameters)
@rule Observation(:out,Marginalisation) (m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::PSubstitutionMeta) = begin
    A = mean(q_A)
    s = probvec(q_in)
    return Dirichlet(A * s .+ 1.0)
end
