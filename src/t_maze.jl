using Pkg;Pkg.activate(".");Pkg.instantiate()
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions
enable_autocomplete_brackets(false)
include("transition_mixture.jl")
include("categorical.jl")

softmax(x) = exp.(x) ./ sum(exp.(x))

# Stolen from the Epistemic Value paper
function constructABCD(α::Float64, c::Float64)
    # Observation model
    A_1 = [0.5 0.5;
           0.5 0.5;
           0.0 0.0;
           0.0 0.0]

    A_2 = [0.0 0.0;
           0.0 0.0;
           α   1-α;
           1-α α  ]

    A_3 = [0.0 0.0;
           0.0 0.0;
           1-α α  ;
           α   1-α]

    A_4 = [1.0 0.0;
           0.0 1.0;
           0.0 0.0;
           0.0 0.0]

    A = zeros(16, 8)
    A[1:4, 1:2]   = A_1
    A[5:8, 3:4]   = A_2
    A[9:12, 5:6]  = A_3
    A[13:16, 7:8] = A_4

    # Transition model
    B_1 = kron([1 0 0 1; # Row: can I move to 1?
                0 1 0 0;
                0 0 1 0;
                0 0 0 0], I(2))

    B_2 = kron([0 0 0 0;
                1 1 0 1; # Row: can I move to 2?
                0 0 1 0;
                0 0 0 0], I(2))

    B_3 = kron([0 0 0 0;
                0 1 0 0;
                1 0 1 1; # Row: can I move to 3?
                0 0 0 0], I(2))

    B_4 = kron([0 0 0 0;
                0 1 0 0;
                0 0 1 0;
                1 0 0 1], I(2)) # Row: can I move to 4?

    B = [B_1, B_2, B_3, B_4]

    # Goal prior
    C = softmax(kron(ones(4), [0.0, 0.0, c, -c]))

    # Initial state prior
    # Note: in the epistemic value paper (Friston, 2015) there is a softmax over D.
    # However, from the context as described in the paper this appears to be a notational error.
    D = kron([1.0, 0.0, 0.0, 0.0], [0.5, 0.5])

    return (A, B, [C,C], D)
end

A,B,C,D = constructABCD(0.9,2.)

@model function t_maze(A,D,B1,B2,B3,B4)
    z_0 ~ Categorical(D)

    z = randomvar(2)
    switch = randomvar(2)

    x = datavar(Vector{Float64}, 2)
    z_prev = z_0

    for t in 1:2
	switch[t] ~ Categorical(fill(1. /4. ,4))
	z[t] ~ TransitionMixture(z_prev,switch[t], B1,B2,B3,B4)
	x[t] ~ GFECategorical(z[t], A) where {pipeline=RequireInbound(in = Categorical(fill(1. /8. ,8)))}
        z_prev = z[t]
    end
end

imodel = Model(t_maze,A,D,B[1],B[2],B[3],B[4])

result = inference(model = imodel, data= (x = C,))

probvec(result.posteriors[:switch][1][1])


