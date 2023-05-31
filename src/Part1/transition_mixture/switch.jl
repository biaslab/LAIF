import Base.Iterators.repeated

#@rule TransitionMixture{N}(:switch, Marginalisation) (m_in::Categorical,m_out::Categorical,q_B::ManyOf{Any,N}) where {N} = begin
#@rule TransitionMixture(:z, Marginalisation) (m_in::Categorical,m_out::Categorical,q_B1::Union{MatrixDirichlet,PointMass},q_B2::Union{MatrixDirichlet,PointMass}, q_B3::Union{MatrixDirichlet,PointMass}, q_B4::Union{MatrixDirichlet,PointMass} ) = begin
@rule TransitionMixture(:z, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
    #q_Bs = mean.(q_B)      # Transition matrices
    q_Bs = [mean(q_B1),mean(q_B2),mean(q_B3),mean(q_B4)]      # Transition matrices
    outp = probvec(m_out)  # Output
    inp = probvec(m_in)    # input

    #p = mapreduce(x -> x[1]' * x[2] * x[3], +, zip(repeated(outp), q_Bs, repeated(inp)))
    p = map(x -> x[1]' * x[2] * x[3],  zip(repeated(outp), q_Bs, repeated(inp)))
    return Categorical(p ./ sum(p))
end

# Used when the initial state is fixed
@rule TransitionMixture(:z, Marginalisation) (m_out::DiscreteNonParametric, q_in::PointMass, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
    q_Bs = [mean(q_B1),mean(q_B2),mean(q_B3),mean(q_B4)]      # Transition matrices
    outp = probvec(m_out)  # Output
    inp = probvec(q_in)    # input

    #p = mapreduce(x -> x[1]' * x[2] * x[3], +, zip(repeated(outp), q_Bs, repeated(inp)))
    p = map(x -> x[1]' * x[2] * x[3],  zip(repeated(outp), q_Bs, repeated(inp)))
    return Categorical(p ./ sum(p))
end
