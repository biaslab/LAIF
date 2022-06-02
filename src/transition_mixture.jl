#using Pkg;Pkg.activate(".");Pkg.instantiate()
#using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL
#enable_autocomplete_brackets(false)
#using ReactiveMP:Categorical


struct TransitionMixture end

@node TransitionMixture Stochastic [in,out,z,B1,B2,B3,B4,]

# m_x means message_x, q_x means marginal x. So we use this to dispatch on SP/VB rules
@rule TransitionMixture(:out, Marginalisation) (m_in::Categorical,m_z::Categorical,q_B1::PointMass,q_B2::PointMass,q_B3::PointMass,q_B4::PointMass,) = begin
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

@rule TransitionMixture(:in, Marginalisation) (m_out::Categorical,m_z::Categorical,q_B1::PointMass,q_B2::PointMass,q_B3::PointMass,q_B4::PointMass,) = begin
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

@rule TransitionMixture(:z, Marginalisation) (m_in::Categorical,m_out::Categorical,q_B1::PointMass,q_B2::PointMass,q_B3::PointMass,q_B4::PointMass,) = begin

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
	p[k] += inp' * Bs[k] * out
	#p[k] += out' * Bs[k] * inp
    end

    return Categorical(p ./ sum(p))
end
