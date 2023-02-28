
using ForwardDiff: jacobian
using DomainSets: FullSpace
include("function.jl")

# Helper functions
# We don't want log(0) to happen
safelog(x) = log(clamp(x,tiny,Inf))
softmax(x) = exp.(x) ./ sum(exp.(x))

@average_energy Transition (q_out::Categorical, q_in::Categorical, q_A::Union{MatrixDirichlet,PointMass},meta::EpistemicMeta) = begin
    s = probvec(q_in)
    A = mean(q_A)
    c = probvec(q_out)

    -s' * diag(A' * safelog.(A)) - (A*s)'*safelog.(c)
end


struct EpistemicMeta end

@rule Transition(:out, Marginalisation) (q_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin
    a = clamp.(exp.(mean(log, q_a) * probvec(q_in)), tiny, Inf)
    μ = Categorical(a ./ sum(a))
    return (A = q_a, in = q_in, μ = μ)
end

@rule Transition(:a, Marginalisation) (m_out::NamedTuple, m_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin
    A_bar = mean(q_a)
    c = mean(m_out[:out]) # This comes from the `Categorical` node as a named tuple
    s = probvec(m_in)
    # LogPdf
    logpdf(A) = s' *( diag(A'*safelog.(A)) + A'*(safelog.(c) - safelog.(A_bar*s)))
    return ContinuousMatrixvariateLogPdf(FullSpace(),logpdf)
end

@rule Transition(:in, Marginalisation) (m_out::Any, m_in::Categorical, q_in::Categorical, q_a::Any, meta::EpistemicMeta) = begin

    s = probvec(q_in)
    d = probvec(m_in)
    A = mean(q_a)

    # We use the goal prior on an edge here
    C = mean(m_out[:out])

    # Newton iterations for stability
    g(s) = s - softmax(safelog.(d) + diag(A' * safelog.(A)) + A' *(safelog.(C) - safelog.(A * s)))

    s_k = deepcopy(s)
    for i in 1:20 # TODO make this user specified
        s_k = s_k - inv(jacobian(g,s_k)) * g(s_k)
    end

    ρ = s_k ./ (d .+ 1e-6)
    return Categorical(ρ ./ sum(ρ))
end

@rule Categorical(:out, Marginalisation) (q_p::Any, meta::EpistemicMeta) = begin
    return (p = q_p, )
end

@rule Categorical(:p, Marginalisation) (m_out::NamedTuple, meta::EpistemicMeta) = begin
    A = mean(m_out[:A])
    s = probvec(m_out[:in])
    return Dirichlet(A * s .+ 1.0)
end

