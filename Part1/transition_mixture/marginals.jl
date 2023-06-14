
#@marginalrule TransitionMixture(:out_in_z) (m_out::DiscreteNonParametric, m_z::DiscreteNonParametric, m_in::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
@marginalrule TransitionMixture(:out_in_z) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin

    μ_out = probvec(m_out)
    μ_in = probvec(m_in)
    μ_z = probvec(m_z)

    # Need to make this generic
    A_tilde = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]

    B = cat(map( x -> x * μ_out * μ_in', μ_z)..., dims=3) .* A_tilde
    return ContingencyTensor(B ./ sum(B))
end

@marginalrule TransitionMixture(:out_z) (m_out::DiscreteNonParametric, m_z::DiscreteNonParametric, q_in::PointMass, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
    @call_marginalrule TransitionMixture(:out_in_z) (m_out = m_out, m_z = m_z, m_in = Categorical(mean(q_in)), q_B1 = q_B1, q_B2 = q_B2, q_B3 = q_B3, q_B4 = q_B4, )
end

