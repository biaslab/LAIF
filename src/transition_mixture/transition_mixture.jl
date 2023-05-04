@marginalrule TransitionMixture(:out_in_z) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin

    I_ = length(probvec(m_out))
    J  = length(probvec(m_in))
    K  = length(probvec(m_z))
    μ_out = probvec(m_out)
    μ_in = probvec(m_in)
    μ_z = probvec(m_z)

    A_tilde = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]
    B = zeros(I_,J,K)
    for i in 1:I_
        for j in 1:J
            for k in 1:K
                B[i,j,k] = μ_out[i]*μ_in[j]*μ_z[k] * A_tilde[i,j,k]
            end
        end
    end
    return ContingencyTensor(B ./ sum(B))
end

@average_energy TransitionMixture (q_out_in_z::ContingencyTensor, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, meta::PSubstitutionMeta) = begin

    log_A_bar = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]

    U = 0
    for i in 1:4
       U+= -tr(B[:,:,i]' *log_A_bar[:,:,i] )
    end
    U
end

include("in.jl")
include("out.jl")
include("switch.jl")
include("marginals.jl")

