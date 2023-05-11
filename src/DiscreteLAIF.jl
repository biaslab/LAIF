using ForwardDiff: jacobian
using DomainSets: FullSpace
include("matrixlogpdf.jl")

#TODO: make sure BetheMeta does not take any arbitrary thing
#TODO: Reproduced FL experiments
#TODO: Check that pipeline works for messages towards A. Do we still get q_A?

# Use custom pipeline to get marginal and message on incoming edge
struct GFEPipeline{I,M} <: ReactiveMP.AbstractNodeFunctionalDependenciesPipeline
    indices::I
    init_message::M
end

function ReactiveMP.message_dependencies(pipeline::GFEPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # We simply override the messages dependencies with the provided indices
    for index in pipeline.indices
        output = ReactiveMP.messagein(nodeinterfaces[index])
        ReactiveMP.setmessage!(output, pipeline.init_message[index])
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
safelog(x) = ReactiveMP.clamplog(x)#log(clamp(x,tiny,Inf))
softmax(x) = exp.(x) ./ sum(exp.(x))

# Metas to change behaviour between BFE and GFE
struct PSubstitutionMeta end
struct UnobservedBetheMeta end

# Optionally add data constraints
struct ObservedBetheMeta{I}
    δ::I
end

struct ObservedPSubstitutionMeta{I}
    δ::I
end


struct DiscreteLAIF end
# Define node
@node DiscreteLAIF Stochastic [out,in, A]

### Unobserved PSubstitution

# Compute the h vector
_h(A) = -diag(A' * ReactiveMP.clamplog.(A))
mean_h(A) = mean( _h.(rand(A,50)))


# GFE Energy
@average_energy DiscreteLAIF (q_out::Union{Dirichlet,PointMass}, q_in::Categorical, q_A::Union{PointMass,MatrixDirichlet},meta::PSubstitutionMeta) = begin
    s = probvec(q_in)
    A = mean(q_A)
    c = mean(q_out)

    #-s' * diag(A' * safelog.(A)) + (A*s)'*(safelog.(A*s) .- safelog.(c))
    -s' * -mean_h(q_A) + (A*s)'*(safelog.(A*s) .- safelog.(c))
end


# GFE Message towards A
@rule DiscreteLAIF(:A, Marginalisation) (q_out::Union{PointMass,Dirichlet}, m_in::Categorical, q_in::Categorical, q_A::MatrixDirichlet, m_A::Any, meta::PSubstitutionMeta) = begin
    A_bar = mean(q_A)
    c = mean(q_out)
    s = probvec(q_in)

    # LogPdf
    logpdf(A) = s' *( diag(A'*safelog.(A)) + A'*(safelog.(c) - safelog.(A_bar*s)))
    return ContinuousMatrixvariateLogPdf(FullSpace(),logpdf)
end

# GFE Message towards input
@rule DiscreteLAIF(:in, Marginalisation) (q_out::Union{Dirichlet,PointMass},m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass}, m_A::Any,meta::PSubstitutionMeta) = begin
    # Ignore message from A
    @call_rule typeof(DiscreteLAIF)(:in,Marginalisation) (q_out = q_out, m_in = m_in, q_in = q_in, q_A = q_A, meta=meta)
end


@rule DiscreteLAIF(:in, Marginalisation) (m_in::DiscreteNonParametric, q_out::Union{Dirichlet,PointMass}, q_in::DiscreteNonParametric, q_A::Union{MatrixDirichlet,PointMass}, meta::PSubstitutionMeta) = begin

    d = probvec(m_in)
    s = probvec(q_in)
    A = mean(q_A)
    C = mean(q_out)

    # Newton iterations for stability
    #g(s) = s - softmax(diag(A'*safelog.(A)) + A'*safelog.(C) - A'*safelog.(A*s) + safelog.(d))
    g(s) = s - softmax( -mean_h(q_A) + A'*safelog.(C) - A'*safelog.(A*s) + safelog.(d))

    s_k = deepcopy(s)
    for i in 1:20 # TODO make this user specified
        s_k = s_k - inv(jacobian(g,s_k)) * g(s_k)
    end

    ρ = s_k ./ (d .+ 1e-6)

    return Categorical(ρ ./ sum(ρ))
end


# Message towards output (goal parameters)
@rule DiscreteLAIF(:out,Marginalisation) (m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::PSubstitutionMeta) = begin
    A = mean(q_A)
    s = probvec(q_in)
    return Dirichlet(A * s .+ 1.0)
end

### Observed PSubstitution
# GFE Energy
@average_energy DiscreteLAIF (q_out::Union{Dirichlet,PointMass}, q_in::Categorical, q_A::Union{PointMass,MatrixDirichlet},meta::ObservedPSubstitutionMeta) = begin
    x = meta.δ
    s = probvec(q_in)
    A = mean(q_A)
    c = mean(q_out)

    -x' * (safelog.(A)*s + safelog.(c))
end


# GFE Message towards A
@rule DiscreteLAIF(:A, Marginalisation) (q_out::Union{PointMass,Dirichlet}, m_in::Categorical, q_in::Categorical, q_A::MatrixDirichlet, meta::ObservedPSubstitutionMeta) = begin
    x = meta.δ
    s = probvec(q_in)

    return MatrixDirichlet(x*s' .+ 1)
end

# GFE Message towards input
@rule DiscreteLAIF(:in, Marginalisation) (q_out::Union{Dirichlet,PointMass},m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::ObservedPSubstitutionMeta) = begin
    x = meta.δ
    A = mean(q_A)
    ρ = safelog.(A)' * x

    return Categorical(softmax(ρ))
end


# Message towards output (goal parameters)
@rule DiscreteLAIF(:out,Marginalisation) (m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::ObservedPSubstitutionMeta) = begin
    x = meta.δ
    return Dirichlet(x .+ 1)
end

### Unobserved Bethe
# BFE Energy
@average_energy DiscreteLAIF (q_out::Union{Dirichlet,PointMass}, q_in::Categorical, q_A::Union{PointMass,MatrixDirichlet},meta::UnobservedBetheMeta) = begin

    s = probvec(q_in)
    A = mean(q_A)
    c = mean(q_out)

    # Compute internal marginal
    x = softmax(safelog.(A) * s + safelog.(c))

    -x' * (safelog.(A)*s + safelog.(c)) + x' * safelog.(x)
end


# BFE Message towards A
@rule DiscreteLAIF(:A, Marginalisation) (q_out::Union{PointMass,Dirichlet}, m_in::Categorical, q_in::Categorical, q_A::MatrixDirichlet, meta::UnobservedBetheMeta) = begin

    c = mean(q_out)
    s = probvec(q_in)
    A = mean(q_A)

    # Compute internal marginal
    x = softmax(safelog.(A) * s + safelog.(c))

    return MatrixDirichlet(x*s' .+ 1)
end

# BFE Message towards input
@rule DiscreteLAIF(:in, Marginalisation) (q_out::Union{Dirichlet,PointMass},m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::UnobservedBetheMeta) = begin
    c = mean(q_out)
    s = probvec(q_in)
    A = mean(q_A)

    # Compute internal marginal
    x = softmax(safelog.(A) * s + safelog.(c))

    ρ = safelog.(A)' * x

    return Categorical(softmax(ρ))
end


# BFE Message towards output (goal parameters)
@rule DiscreteLAIF(:out,Marginalisation) (m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::UnobservedBetheMeta) = begin
    c = mean(q_out)
    s = probvec(q_in)
    A = mean(q_A)

    # Compute internal marginal
    x = softmax(safelog.(A) * s + safelog.(c))
    return Dirichlet(x .+ 1)
end

### Observed Bethe
# BFE Energy
@average_energy DiscreteLAIF (q_out::Union{Dirichlet,PointMass}, q_in::Categorical, q_A::Union{PointMass,MatrixDirichlet},meta::ObservedBetheMeta) = begin

    s = probvec(q_in)
    A = mean(q_A)
    c = mean(q_out)

    x = meta.δ

    -x' * (safelog.(A)*s + safelog.(c)) + x' * safelog.(x)
end


# BFE Message towards A
@rule DiscreteLAIF(:A, Marginalisation) (q_out::Union{PointMass,Dirichlet}, m_in::Categorical, q_in::Categorical, q_A::MatrixDirichlet, meta::ObservedBetheMeta) = begin

    s = probvec(q_in)
    x = meta.δ

    return MatrixDirichlet(x*s' .+ 1)
end

# BFE Message towards input
@rule DiscreteLAIF(:in, Marginalisation) (q_out::Union{Dirichlet,PointMass},m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::ObservedBetheMeta) = begin
    A = mean(q_A)
    x = meta.δ

    ρ = safelog.(A)' * x

    return Categorical(softmax(ρ))
end


# BFE Message towards output (goal parameters)
@rule DiscreteLAIF(:out,Marginalisation) (m_in::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::ObservedBetheMeta) = begin
    x = meta.δ
    return Dirichlet(x .+ 1)
end
