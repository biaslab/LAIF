#using Pkg; Pkg.activate("..");Pkg.instantiate()
#using RxInfer,ReactiveMP,OhMyREPL,Distributions
#enable_autocomplete_brackets(false);colorscheme!("GruvboxDark")

import Base.Iterators.repeated

@rule TransitionMixture{N}(:in, Marginalisation) (q_switch::Categorical,q_out::Categorical,q_B::ManyOf{Any,N}) where {N} = begin
    πs = probvec(q_switch) # Weights
    q_Bs = mean.(q_B)      # Transition matrices
    outp = probvec(q_out)  # Output

    p = mapreduce(x -> x[1] * x[2]' * x[3], +, zip(πs, q_Bs,repeated(outp)))
    return Categorical(p ./ sum(p))
end
