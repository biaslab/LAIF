struct TransitionMixture end

@node TransitionMixture Stochastic [out,in,z,B1,B2,B3,B4,]

# Average energy functional
#function averageEnergy(::Type{TransitionMixture},
#                       dist_out_in1_switch::Distribution{Multivariate, Contingency},
#                       dist_factors::Vararg{Distribution})
#
#    n_factors = length(dist_factors)
#    U = 0.0
#    for k = 1:n_factors
#        U += -tr(dist_out_in1_switch.params[:p][k]'*unsafeLogMean(dist_factors[k]))
#    end
#
#    return U
#end

#@average_eneergy TransitionMixture(blah) = begin
#    TODO
#end


# m_x means message_x, q_x means marginal x. So we use this to dispatch on SP/VB rules
@rule TransitionMixture(:out, Marginalisation) (m_in::DiscreteNonParametric,m_z::DiscreteNonParametric,q_B1::PointMass,q_B2::PointMass,q_B3::PointMass,q_B4::PointMass,) = begin
    z = probvec(m_z)

    # Hacky McHackface
    B1 = mean(q_B1)
    B2 = mean(q_B2)
    B3 = mean(q_B3)
    B4 = mean(q_B4)
    Bs = [B1,B2,B3,B4]

    inp = probvec(m_in)

    # Hack some more...
    p = zeros(size(B1*inp)[1])
    for k in 1:4
	p += z[k] * Bs[k] * inp
    end

    return Categorical(p ./ sum(p))
end

@rule TransitionMixture(:in, Marginalisation) (m_out::DiscreteNonParametric,m_z::DiscreteNonParametric,q_B1::PointMass,q_B2::PointMass,q_B3::PointMass,q_B4::PointMass,) = begin
    z = probvec(m_z)

    # Hacky McHackface
    B1 = mean(q_B1)
    B2 = mean(q_B2)
    B3 = mean(q_B3)
    B4 = mean(q_B4)
    Bs = [B1,B2,B3,B4]

    out = probvec(m_out)

    # This is ugly..
    p = zeros(size(B1'*out)[1])
    for k in 1:4
	p += z[k] * Bs[k]' * out
    end

    return Categorical(p ./ sum(p))
end

@rule TransitionMixture(:z, Marginalisation) (m_out::DiscreteNonParametric,m_in::DiscreteNonParametric,q_B1::PointMass,q_B2::PointMass,q_B3::PointMass,q_B4::PointMass,) = begin

    # Hacky McHackface
    B1 = mean(q_B1)
    B2 = mean(q_B2)
    B3 = mean(q_B3)
    B4 = mean(q_B4)
    Bs = [B1,B2,B3,B4]

    out = probvec(m_out)
    inp = probvec(m_in)

    p = zeros(4)
    # This is out' * B * inp in Thijs implementation?
    for k in 1:4
	#p[k] += inp' * Bs[k] * out
	p[k] += out' * Bs[k] * inp
    end

    return Categorical(p ./ sum(p))
end

