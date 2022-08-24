using ForwardDiff: jacobian

# Helper functions
# We don't want log(0) to happen
safelog(x) = log(x + eps())
#safelog(x) = log(x + exp(-16))
normalize(x) = x ./ sum(x)
#softmax(x) =  exp.(x) ./ sum(exp.(x))

struct GFECategorical end
@node GFECategorical Stochastic [out,in,A]

@average_energy GFECategorical (q_out::PointMass, q_in::Categorical,q_A::PointMass,) = begin
    σ = probvec(q_in)
    A = mean(q_A)
    c = probvec(q_out)

    -σ' * diag(A' * safelog.(A)) + (A*σ)'* (safelog.(A*σ - safelog.(c)))
end

@rule GFECategorical(:in, Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::PointMass, q_A::PointMass,) = begin
    z = probvec(q_in)
    d = probvec(m_in)
    A = mean(q_A)

    # We use the goal prior on an edge here
    C = probvec(q_out)

    # Newton iterations for stability
    g(z) = z - softmax(safelog.(d) + diag(A' * safelog.(A)) + A' *(safelog.(C) .- safelog.(A * z)))
    for i in 1:10 # TODO make this user specified
        z = z - inv(jacobian(g,z)) * g(z)
    end

    ρ = normalize(z ./ clamp.(d,tiny,huge))
    return Categorical(ρ)
end

#@rule GFECategorical(:in, Marginalisation) (q_out::PointMass, m_in::DiscreteNonParametric, q_A::PointMass, meta::GFEMeta,) = begin
#    # We store outgoing messages in meta since we can't access the edge marginal directly
#    μ_in = meta.μ_in
#    q_in = prod(ProdAnalytical(),m_in,μ_in)
#
#    z = probvec(q_in)
#    A = mean(q_A)
#    # We use the goal prior on an edge here
#    C = probvec(q_out)
#    # q_out needs to be A*mean(incoming), hence this line
#    x = A * z
#    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
#    m_out = Categorical(exp.(ρ) / sum(exp.(ρ)))
#
#    meta.μ_in = m_out
#    return m_out
#end
#
#@rule GFECategorical(:in, Marginalisation) (q_out::PointMass, q_in::DiscreteNonParametric,q_A::PointMass) = begin
#    z = probvec(q_in)
#    A = mean(q_A)
#    # We use the goal prior on an edge here
#    C = probvec(q_out)
#    # q_out needs to be A*mean(incoming), hence this line
#    x = A * z
#    ρ = clamp.( exp.(diag(A' * safelog.(A)) + A' *(safelog.(C) .- safelog.(x))), tiny, huge)
#
#    return Categorical(ρ ./ clamp(sum(ρ),tiny,huge))
#end
