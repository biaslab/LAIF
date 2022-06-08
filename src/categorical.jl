struct GFECategorical end
@node GFECategorical Stochastic [out,in,A]

# We don't want log(0) to happen
safelog(x) = log(x +eps())

#note, why is it m_in and not q_in???
@rule GFECategorical(:in, Marginalisation) (q_out::PointMass,q_in::Categorical, q_A::PointMass) = begin
    z = probvec(q_in)
    A = mean(q_A)
    # We use the goal prior on an edge here
    C = probvec(q_out)
    # q_out needs to be A*mean(incoming), hence this line
    x = A * z
    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
    return Categorical(exp.(ρ) / sum(exp.(ρ)))
end

# How do we get q_in??
@rule GFECategorical(:in, Marginalisation) (q_out::PointMass,m_in::Categorical, q_A::PointMass) = begin
    z = probvec(m_in)
    A = mean(q_A)
    # We use the goal prior on an edge here
    C = probvec(q_out)
    # q_out needs to be A*mean(incoming), hence this line
    x = A * z
    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
    return Categorical(exp.(ρ) / sum(exp.(ρ)))
end
