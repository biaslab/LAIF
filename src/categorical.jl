struct GFECategorical end
@node GFECategorical Stochastic [out,in,A]

# We don't want log(0) to happen
safelog(x) = log(x +eps())

@rule GFECategorical(:in, Marginalisation) (q_out::PointMass,m_in::Categorical, q_A::PointMass) = begin
    z = probvec(m_in)
    A = mean(q_A)
    # We use the goal prior on an edge here
    C = probvec(q_out)
    # q_out needs to be A*mean(incoming), hence this line
    x = A * z
    # Write this out in a nicer way
   # ρ = zeros(size(z))
   # for i in 1:size(A)[2]
   #     #ρ[i] = z[i] * A[:,i]' * safelog.(A[:,i]) + z[i] * A[:,i]' * safelog.(C) - z[i] * A[:,i]' * safelog.(x)
   #     # Below is the correct one
   #     #ρ[i] = A[:,i]' * safelog.(A[:,i]) + A[:,i]' * safelog.(C) - A[:,i]' * safelog.(x)
   #     ρ[i] = A[:,i]' * safelog.(A[:,i]) + A[:,i]' * (safelog.(C) .- safelog.(x))
   # end
    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
    return Categorical(exp.(ρ) / sum(exp.(ρ)))
end
