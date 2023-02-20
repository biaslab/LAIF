using Random

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

    # Transition model (with forced move back after reward-arm visit)
    B_1 = kron([1 1 1 1; # Row: can I move to 1?
                0 0 0 0;
                0 0 0 0;
                0 0 0 0], I(2))

    B_2 = kron([0 1 1 0; 
                1 0 0 1; # Row: can I move to 2?
                0 0 0 0;
                0 0 0 0], I(2))

    B_3 = kron([0 1 1 0;
                0 0 0 0;
                1 0 0 1; # Row: can I move to 3?
                0 0 0 0], I(2))

    B_4 = kron([0 1 1 0;
                0 0 0 0;
                0 0 0 0;
                1 0 0 1], I(2)) # Row: can I move to 4?

    B = [B_1, B_2, B_3, B_4]

    # Goal prior
    C = softmax(kron(ones(4), [0.0, 0.0, c, -c]))

    # Initial state prior
    # Note: in the epistemic value paper (Friston, 2015) there is a softmax over D.
    # However, from the context as described in the paper this appears to be a notational error.
    D = kron([1.0, 0.0, 0.0, 0.0], [0.5, 0.5])

    return (A, B, C, D)
end

function generateGoalSequence(S::Int64)
    rs = Vector{Vector}(undef, S)
    for si=1:S
        if rand() > 0.5
            rs[si] = [0, 1]
        else
            rs[si] = [1, 0]
        end
    end

    return rs
end

function generateGoalSequence(seed::Int64, S::Int64)
    Random.seed!(seed)
    generateGoalSequence(S)
end


function initializeWorld(A, B, C, D, rs)
    function reset(s)
       x_0 = zeros(8)
       x_0[1:2] = rs[s]
       x_t_min = x_0
       o_t = A*x_0

       return Int64(r'*[2, 3]) # Hidden reward position
    end

    # Set reward position
    r = [0, 1]

    # Initial state
    x_0 = zeros(8)
    x_0[1:2] = r # Start from position 1

    # Execute a move to position a_t
    x_t_min = x_0
    function execute(a_t::Int64)
        x_t = B[a_t]*x_t_min # State transition
        o_t = A*x_t # Observation

        x_t_min = x_t # Reset state for next step
    end

    o_t = A*x_0
    observe() = sample(Distribution(Univariate, Categorical, p=o_t))

    return (reset, execute, observe)
end
;