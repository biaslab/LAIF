using ForwardDiff: jacobian
using DomainSets: FullSpace
include("function.jl")

# Helper functions
# We don't want log(0) to happen
safelog(x) = log(clamp(x,tiny,Inf))
softmax(x) = exp.(x) ./ sum(exp.(x))

ReactiveMP.mean(::typeof(safelog), p::PointMass) = safelog.(mean(p))

import ReactiveMP: AbstractFormConstraint

struct EpistemicProduct <: AbstractFormConstraint end

ReactiveMP.make_form_constraint(::Type{EpistemicProduct}) = EpistemicProduct()

ReactiveMP.is_point_mass_form_constraint(::EpistemicProduct) = false
ReactiveMP.default_form_check_strategy(::EpistemicProduct)   = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::EpistemicProduct)       = ProdGeneric()

function ReactiveMP.constrain_form(::EpistemicProduct, distribution)
    error("Unexpected")
end

struct WMarginal{A, I, P, Q}
    m_a :: A # q(A)
    m_in :: I # q(A) * q(z), Ā * z̄
    m_p :: P # Marginal over C
    q :: Q # Edge Marginal
end

ReactiveMP.probvec(dist::WMarginal) = ReactiveMP.probvec(dist.q)
Distributions.entropy(dist::WMarginal) = entropy(dist.q)

function ReactiveMP.constrain_form(::EpistemicProduct, distribution::DistProduct)
    return WMarginal(distribution.left[:A], distribution.left[:in], distribution.right[:p], distribution.left[:μ])
end

struct GFEPipeline{I} <: ReactiveMP.AbstractNodeFunctionalDependenciesPipeline
    indices::I
end

function ReactiveMP.message_dependencies(pipeline::GFEPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # We simply override the messages dependencies with the provided indices
    # for index in pipeline.indices
    #     output = ReactiveMP.messagein(nodeinterfaces[index])
    #     ReactiveMP.setmessage!(output, Categorical(ones(8) ./ 8))
    # end
    return map(inds -> map(i -> @inbounds(nodeinterfaces[i]), inds), pipeline.indices)
end

function ReactiveMP.marginal_dependencies(pipeline::GFEPipeline, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # For marginals we require marginal on the same edges as in the provided indices
    require_marginal = ReactiveMP.RequireMarginalFunctionalDependencies(pipeline.indices, map(_ -> nothing, pipeline.indices))
    return ReactiveMP.marginal_dependencies(require_marginal, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
end

struct EpistemicMeta end

@average_energy Transition (q_out::WMarginal, q_in::Categorical, q_a::PointMass, meta::EpistemicMeta) = begin
    s = probvec(q_in)
    A = mean(q_a)
    c = probvec(q_out)

    -s' * diag(A' * safelog.(A)) - (A*s)'*safelog.(c)
end

@average_energy Categorical (q_out::WMarginal, q_p::Any, meta::EpistemicMeta) = begin
    -sum(probvec(q_out) .* mean(ReactiveMP.clamplog, q_p))
end

@rule Transition(:out, Marginalisation) (m_in::Categorical, q_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin
    a = clamp.(exp.(mean(safelog, q_a) * probvec(q_in)), tiny, Inf)
    μ = Categorical(a ./ sum(a))
    return (A = q_a, in = q_in, μ = μ)
end

@rule Transition(:out, Marginalisation) (m_in::Categorical, q_in::Categorical, q_a::PointMass, meta::EpistemicMeta) = begin
    a = mean(q_a) * probvec(q_in)
    μ = Categorical(a ./ sum(a))
    return (A = q_a, in = q_in, μ = μ)
end

# Message towards A
@rule Transition(:a, Marginalisation) (q_out::WMarginal, m_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin
    A_bar = mean(q_a)
    c = mean(q_out.m_p) # This comes from the `Categorical` node as a named tuple TODO: bvdmitri double check
    s = probvec(q_in)
    # LogPdf
    logpdf(A) = s' *( diag(A'*safelog.(A)) + A'*(safelog.(c) - safelog.(A_bar*s)))
    return ContinuousMatrixvariateLogPdf(FullSpace(),logpdf)
end

#@rule Transition(:a, Marginalisation) (q_out::WMarginal, m_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin
#    A_bar = mean(q_a)
#    c = mean(q_out.m_p) # This comes from the `Categorical` node as a named tuple TODO: bvdmitri double check
#    s = probvec(q_in)
#    # LogPdf
#    logpdf(A) = s' *( diag(A'*safelog.(A)) + A'*(safelog.(c) - safelog.(A_bar*s)))
#    return ContinuousMatrixvariateLogPdf(FullSpace(),logpdf)
#end


@rule Transition(:in, Marginalisation) (q_out::WMarginal, m_in::Categorical, q_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin

    s = probvec(q_in)
    d = probvec(m_in)
    A = mean(q_a)

    # We use the goal prior on an edge here
    C = mean(q_out.m_p) # TODO: bvdmitri double check. Checks out --M

    # Newton iterations for stability
    g(s) = s - softmax(safelog.(d) + diag(A' * safelog.(A)) + A' *(safelog.(C) - safelog.(A * s)))

    s_k = deepcopy(s)
    for i in 1:20 # TODO make this user specified
        s_k = s_k - inv(jacobian(g,s_k)) * g(s_k)
    end

    ρ = s_k ./ (d .+ 1e-6)
    return Categorical(ρ ./ sum(ρ))
end

# Rule towards "observations"
@rule Categorical(:out, Marginalisation) (q_p::Any, meta::EpistemicMeta) = begin
    return (p = q_p, )
end

# Rule towards goal parameters
@rule Categorical(:p, Marginalisation) (q_out::WMarginal, meta::EpistemicMeta) = begin
    A = mean(q_out.m_a)
    s = probvec(q_out.m_in)
    return Dirichlet(A * s .+ 1.0)
end

