struct GFECategorical end

# We use this to keep track of outgoing messages on the :in edge in order to compute the marginals we need
mutable struct GFEMeta
    μ_in::Any
end

@node GFECategorical Stochastic [out,in,A]

@average_energy GFECategorical (q_out::PointMass, q_in::Categorical,q_A::PointMass,) = begin
    σ = probvec(q_in)
    A = mean(q_A)
    c = probvec(q_out)

    -σ' * diag(A' * safelog.(A)) + (A*σ)'* (safelog.(A*σ - safelog.(c)))
end

# We don't want log(0) to happen
safelog(x) = log(x +eps())


# How do we get q_in??
@rule GFECategorical(:in, Marginalisation) (q_out::PointMass, m_in::DiscreteNonParametric, q_A::PointMass, meta::GFEMeta,) = begin
    # We store outgoing messages in meta since we can't access the edge marginal directly
    μ_in = meta.μ_in
    q_in = prod(ProdAnalytical(),m_in,μ_in)

    z = probvec(q_in)
    A = mean(q_A)
    # We use the goal prior on an edge here
    C = probvec(q_out)
    # q_out needs to be A*mean(incoming), hence this line
    x = A * z
    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
    m_out = Categorical(exp.(ρ) / sum(exp.(ρ)))

    meta.μ_in = m_out
    return m_out
end

