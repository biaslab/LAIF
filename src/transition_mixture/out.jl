import Base.Iterators.repeated

@rule TransitionMixture{N}(:out, Marginalisation) (q_switch::Categorical,q_in::Categorical,q_B::ManyOf{Any}) where {N} = begin
    πs = probvec(q_switch) # Weights
    q_Bs = mean.(q_B)      # Transition matrices
    inp = probvec(q_in)    # Input

    W = mapreduce(x -> x[1] * x[2] * x[3], +, zip(πs, q_Bs,repeated(inp)))
    return Categorical(W ./ sum(W))
end
