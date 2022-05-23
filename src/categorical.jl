#using Pkg; Pkg.activate(".");Pkg.instantiate()
#using ReactiveMP, Rocket, GraphPPL, OhMyREPL, LinearAlgebra,Distributions
#enable_autocomplete_brackets(false)

#struct GFECategorical end
#@node GFECategorical Stochastic [in,A,C]
#
#safelog(x) = log(x +eps())
#@rule GFECategorical(:in, Marginalisation) (q_in::Any, q_a::PointMass, q_c::PointMass) = begin
#    z = mean(q_in)
#    A = mean(m_a)
#    C = mean(m_c)
#    # q_out needs to be A*mean(incoming), hence this line
#    x = A * z
#    # Write this out in a nicer way
#    ρ = sum(z .* A .* safelog.(A)',dims= 2) + z.*A * (safelog.(C) - safelog.(x))
#    return Categorical(exp.(ρ) / sum(exp.(ρ)))
#end

struct GFECategorical end
@node GFECategorical Stochastic [out,in,A]

safelog(x) = log(x +eps())
@rule GFECategorical(:in, Marginalisation) (q_out::PointMass,m_in::Categorical, q_A::PointMass) = begin
    z = probvec(m_in)
    A = mean(q_A)
    C = probvec(q_out)
    # q_out needs to be A*mean(incoming), hence this line
    x = A * z
    # Write this out in a nicer way. Vec is there to force the type to not be Matrix
    ρ = vec(sum(z .* A .* safelog.(A)',dims= 2)) + z.*A * (safelog.(C) - safelog.(x))
    return Categorical(exp.(ρ) / sum(exp.(ρ)))
end


# Random crap for testing

#A * z * x' * safelog.(A) * z
#A * z .* (safelog.(C) - safelog.(x))
#
#
#softmax(ρ)
#
#softmax(x) = exp.(x) / sum(exp.(x))
#softmax(zz)
#z = [0,0.5,0.5]
#A = diageye(3)
#A = [0 1 0; 0.5 0. 0.3 ; 0.5 0 0.7]
#x = A * z
#
##@rule Transition(:out, Marginalisation,GFE) (q_in::Any, m_a::PointMass) = begin
##    p = mean(m_a) * probvec(q_in)
##    normalize!(p, 1)
##    return Categorical(p)
##end
#z.*A * (safelog.(Categorical(C)) - safelog.(x))
#safelog(Categorical(C))
#
#A
#C = [0,1,0]
#
#bro = safelog.(C) - safelog.(x)
#
#i = 1
#for i in 1:3
#    ρ_t = z[i] * A'[:,i]' * safelog.(A[:,i]) + (z[i] * A'[:,i]' * (safelog.(C) - safelog.(x)))
#    #println((z[i] * A'[:,i]'* (safelog.(C) - safelog.(x))))
#    zz[i] = ρ_t
#end
#zz
