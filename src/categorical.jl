using ForwardDiff: jacobian

# Helper functions
# We don't want log(0) to happen
safelog(x) = log(clamp(x,tiny,Inf))
#safelog(x) = log(x + exp(-16))
normalize(x) = x ./ sum(x)
#softmax(x) =  exp.(x) ./ sum(exp.(x))

struct GFECategorical end
@node GFECategorical Stochastic [out,in,A]

@average_energy GFECategorical (q_out::PointMass, q_in::Categorical,q_A::PointMass,) = begin
    s = probvec(q_in)
    A = mean(q_A)
    c = probvec(q_out)

    -s' * diag(A' * safelog.(A)) + (A*s)'* safelog.(A*s) - (A*s)'*safelog.(c)
end
# z = s, d = d

@rule GFECategorical(:in, Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::PointMass, q_A::PointMass,) = begin

    z = probvec(q_in)
    d = probvec(m_in)
    A = mean(q_A)

    # We use the goal prior on an edge here
    C = probvec(q_out)

    # Newton iterations for stability
    g(s) = s - softmax(safelog.(d) + diag(A' * safelog.(A)) + A' *(safelog.(C) - safelog.(A * s)))

    z_min = z
    for i in 1:20 # TODO make this user specified
        z_k = z_min - inv(jacobian(g,z_min)) * g(z_min)
        z_min = z_k
    end
    z_k = z_min

    ρ = z_k ./ (d .+ tiny)
    return Categorical(ρ ./ sum(ρ))
end


#@rule GFECategorical(:in, Marginalisation) (m_out::DiscreteNonParametric, q_out::DiscreteNonParametric,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::PointMass, q_A::PointMass,) = begin
#
#    z_0 = probvec(q_in)
#    d = probvec(m_in)
#    A = mean(q_A)
#
#    # We use the goal prior on an edge here
#    C = probvec(q_out)
#
#    # Newton iterations for stability
#    g(z) = z - softmax(safelog.(d) + diag(A' * safelog.(A)) + A' *(safelog.(C) .- safelog.(A * z)))
#    z_min = z_0
#    for i in 1:20 # TODO make this user specified
#        z_k = z_min - inv(jacobian(g,z_min)) * g(z_min)
#        z_min = z_k
#    end
#    z_k = z_min
#
#    ρ = z_k ./ (d .+ tiny)
#    return Categorical(ρ ./ sum(ρ))
#end

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
