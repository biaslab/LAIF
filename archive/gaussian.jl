struct GFEGaussian end
@node GFEGaussian Stochastic [out,in,A]

@average_energy GFEGaussian (q_out::Gaussian, q_in::Gaussian,Σ::PointMass) = begin
    # TODO
end

# We don't want log(0) to happen
safelog(x) = log(x +eps())

#note, why is it m_in and not q_in???
@rule GFEGaussian(:in, Marginalisation) (q_out::Gaussian,q_in::Gaussian,Σa::PointMass) = begin

    # Goal prior
    mc,Σc = meancov(q_out)
    # Incoming marginal
    mx,Σx = meancov(q_in)





    z = probvec(q_in)
    A = mean(q_A)
    # We use the goal prior on an edge here
    C = probvec(q_out)
    # q_out needs to be A*mean(incoming), hence this line
    x = A * z
    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
    return Gaussian(exp.(ρ) / sum(exp.(ρ)))
end

##
##@rule GFEGaussian(:in, Marginalisation) (q_out::PointMass, m_in::DiscreteNonParametric, q_A::PointMass, ) = begin
#
#    z = probvec(m_in)
#    A = mean(q_A)
#    # We use the goal prior on an edge here
#    C = probvec(q_out)
#    # q_out needs to be A*mean(incoming), hence this line
#    x = A * z
#    ρ = diag(A' * safelog.(A)) + A' * (safelog.(C) .- safelog.(x))
#    return Gaussian(exp.(ρ) / sum(exp.(ρ)))
#end

