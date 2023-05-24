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
       z_0 = zeros(8)
       z_0[1:2] = rs[s]
       z_t_min = z_0
       o_t = A*z_0

       return Int64(r'*[2, 3]) # Hidden reward position
    end

    # Set reward position
    r = [0, 1]

    # Initial state
    z_0 = zeros(8)
    z_0[1:2] = r # Start from position 1

    # Execute a move to position a_t
    z_t_min = z_0
    function execute(a_t::Int64)
        z_t = B[a_t]*z_t_min # State transition
        o_t = A*z_t # Observation

        z_t_min = z_t # Reset state for next step
    end

    o_t = A*z_0
    observe() = sample(Distribution(Univariate, Categorical, p=o_t))

    return (reset, execute, observe)
end
;