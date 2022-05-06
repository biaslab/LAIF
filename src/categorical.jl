using Pkg; Pkg.activate(".");Pkg.instantiate()
using ReactiveMP, Rocket, GraphPPL, OhMyREPL, LinearAlgebra
enable_autocomplete_brackets(false)


#@rule Categorical(:out,Marginalisation) (m_in::Categorical, m_a::PointMass ) = begin
#
#    A*x .* (-safelog.(s) + (x' * safelog.(A))' + safelog.(C)))
#    # good stuff goes here
#    return # EFE message
#end

safelog(x) = log(x +eps())
@rule Transition(:in, Marginalisation,GFE) (q_out::Any, m_a::PointMass) = begin

    A = mean(m_a)
    x = mean(q_out)
    C = inbound message on :out
    p = A*x .* (-safelog.(s) + (x' * safelog.(A))' + safelog.(C))

    return Categorical(exp.(p) / sum(exp.(p)))
end


A = [0 1 0; 0.5 0. 0.3 ; 0.5 0 0.7]

x = [0.3,0.2,0.5]
C = [0,1.0,0.0]
s = [.4,0.3,0.3]

norm(A*x .* (-safelog.(s) + (x' * safelog.(A))' + safelog.(C)))

ν = A*x .* (log.(C .+ eps()) - log.(A*s .+eps()) +sum(diagm(x) * log.(A .+ eps())',dims=2))
exp.(ν)/sum(exp.(ν))



