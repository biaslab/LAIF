import Base.Iterators.repeated

#@rule TransitionMixture{N}(:out, Marginalisation) (q_z::Categorical,q_in::Categorical,q_B::ManyOf{Any}) where {N} = begin
@rule TransitionMixture(:out, Marginalisation) (m_in::Union{DiscreteNonParametric, PointMass}, m_z::DiscreteNonParametric, q_B1::PointMass , q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
    πs = probvec(m_z) # Weights
   # q_Bs = mean.(q_B)      # Transition matrices
    q_Bs = [mean(q_B1),mean(q_B2),mean(q_B3),mean(q_B4)]      # Transition matrices
    inp = probvec(m_in)    # Input

    W = mapreduce(x -> x[1] * x[2] * x[3], +, zip(πs, q_Bs,repeated(inp)))
    return Categorical(W ./ sum(W))
end

# Used when the input is fixed
@rule TransitionMixture(:out, Marginalisation) (q_in::PointMass, m_z::DiscreteNonParametric, q_B1::PointMass , q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
    πs = probvec(m_z) # Weights
    q_Bs = [mean(q_B1),mean(q_B2),mean(q_B3),mean(q_B4)]      # Transition matrices
    inp = probvec(q_in)    # Input

    W = mapreduce(x -> x[1] * x[2] * x[3], +, zip(πs, q_Bs,repeated(inp)))
    return Categorical(W ./ sum(W))
end

