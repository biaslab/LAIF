using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL
enable_autocomplete_brackets(false)

using Rocket, ReactiveMP, GraphPPL
using Random, BenchmarkTools, Distributions, LinearAlgebra
using Plots

function rand_vec(rng, distribution::Categorical)
    k = ncategories(distribution)
    s = zeros(k)
    s[ rand(rng, distribution) ] = 1.0
    s
end

function generate_data(n_samples; seed = 124)

    rng = MersenneTwister(seed)

    # Transition probabilities (some transitions are impossible)
    A = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9]
    # Observation noise
    B = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]
    # Initial state
    s_0 = [1.0, 0.0, 0.0]
    # Generate some data
    s = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the states
    x = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the observations

    s_prev = s_0

    for t = 1:n_samples
        a = A * s_prev
        s[t] = rand_vec(rng, Categorical(a ./ sum(a)))
        b = B * s[t]
        x[t] = rand_vec(rng, Categorical(b ./ sum(b)))
        s_prev = s[t]
    end

    return x, s
end

# Model specification
@model function hidden_markov_model(n,Ac,Bc)

    #A ~ MatrixDirichlet(ones(3, 3))
    A = constvar(Ac)

    #B ~ MatrixDirichlet([ 10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0 ])
    B = constvar(Bc)

    s_0 ~ Categorical(fill(1.0 / 3.0, 3))

    s = randomvar(n)
    x = datavar(Vector{Float64}, n)

    s_prev = s_0

    for t in 1:n
        s[t] ~ Transition(s_prev, A)
        x[t] ~ Transition(s[t], B)
        s_prev = s[t]
    end

end

@constraints function hidden_markov_model_constraints()
    q(s_0, s) = q(s_0, s)
end

N = 100
A = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9]
B = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]

x_data, s_data = generate_data(N)

idata = (x = x_data, )

imodel = Model(hidden_markov_model, N,A,B)

imarginals = (
    s = vague(Categorical, 3),
)

ireturnvars = (
    s = KeepLast(),
)

result = inference(
    model         = imodel,
    data          = idata,
    #constraints   = hidden_markov_model_constraints(),
    initmarginals = imarginals,
    returnvars    = ireturnvars,
    iterations    = 20,
    free_energy   = true
)
