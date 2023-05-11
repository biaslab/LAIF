import Base.Iterators.repeated

#@rule TransitionMixture{N}(:out, Marginalisation) (q_z::Categorical,q_in::Categorical,q_B::ManyOf{Any}) where {N} = begin
@rule TransitionMixture(:out, Marginalisation) (m_in::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass , q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
#    return ...
#end
#@rule TransitionMixture(:out, Marginalisation) (m_z::Union{Categorical,PointMass},m_in::Union{Categorical, PointMass} ,q_B1::Union{MatrixDirichlet,PointMass}, q_B2::Union{MatrixDirichlet,PointMass}, q_B3::Union{MatrixDirichlet,PointMass}, q_B4::Union{MatrixDirichlet,PointMass},) = begin
    πs = probvec(m_z) # Weights
   # q_Bs = mean.(q_B)      # Transition matrices
    q_Bs = [mean(q_B1),mean(q_B2),mean(q_B3),mean(q_B4)]      # Transition matrices
    inp = probvec(m_in)    # Input

    W = mapreduce(x -> x[1] * x[2] * x[3], +, zip(πs, q_Bs,repeated(inp)))
    return Categorical(W ./ sum(W))
end
