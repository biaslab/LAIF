export rule

import Base.Iterators.repeated

@rule TransitionMixture{N}(:switch, Marginalisation) (m_in::Categorical,m_out::Categorical,q_B::ManyOf{Any,N}) where {N} = begin
    q_Bs = mean.(q_B)      # Transition matrices
    outp = probvec(m_out)  # Output
    inp = probvec(m_in)    # input

    p = mapreduce(x -> x[1]' * x[2] * x[3], +, zip(repeated(outp), q_Bs, repeated(inp)))
    return Categorical(p ./ sum(p))
end
