import Base.Iterators.repeated

#@rule TransitionMixture{N}(:in, Marginalisation) (q_z::Categorical,q_out::Categorical,q_B::ManyOf{Any,N}) where {N} = begin
#@rule TransitionMixture(:in, Marginalisation) (m_z::Categorical,m_out::Categorical,q_B1::Union{MatrixDirichlet,PointMass}, q_B2::Union{MatrixDirichlet,PointMass}, q_B3::Union{MatrixDirichlet,PointMass}, q_B4::Union{MatrixDirichlet,PointMass} ) = begin
@rule TransitionMixture(:in, Marginalisation) (m_out::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass , q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin

    πs = probvec(m_z) # Weights
    #q_Bs = mean.(q_B)      # Transition matrices
    q_Bs = [mean(q_B1),mean(q_B2),mean(q_B3),mean(q_B4)]      # Transition matrices
    outp = probvec(m_out)  # Output

    p = mapreduce(x -> x[1] * x[2]' * x[3], +, zip(πs, q_Bs,repeated(outp)))
    #p = map(x -> x[1] * x[2]' * x[3], zip(πs, q_Bs,repeated(outp)))
    return Categorical(p ./ sum(p))
end
