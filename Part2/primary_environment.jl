using Random


function constructPrimaryA(α::Float64)
    # Observation model
    A_1 = [0.5 0.5;
           0.5 0.5;
           0.0 0.0;
           0.0 0.0]

    A_2 = [0.0 0.0;
           0.0 0.0;
           α   1-α;
           1-α α  ]

    A_3_α = [0.0 0.0;
             0.0 0.0;
             1-α α  ;
             α   1-α]

    A_3_1 = [0.0 0.0;
             0.0 0.0;
             0.0 1.0
             1.0 0.0]

    A_4 = [1.0 0.0;
           0.0 1.0;
           0.0 0.0;
           0.0 0.0]

    A = zeros(16, 16)
    A[1:4, 1:2]     = A_1
    A[5:8, 3:4]     = A_2
    A[9:12, 5:6]    = A_3_α
    A[13:16, 7:8]   = A_4
    A[1:4, 9:10]    = A_1
    A[5:8, 11:12]   = A_2
    A[9:12, 13:14]  = A_3_1
    A[13:16, 15:16] = A_4

    return A
end

function constructPrimaryBCD(c::Float64)
    # Transition model (with forced move back after reward-arm visit)
    B_1 = kron(I(2), [1 1 1 1; # Row: can I move to 1?
                      0 0 0 0;
                      0 0 0 0;
                      0 0 0 0], I(2))

    B_2 = kron(I(2), [0 1 1 0; 
                      1 0 0 1; # Row: can I move to 2?
                      0 0 0 0;
                      0 0 0 0], I(2))

    B_3 = kron(I(2), [0 1 1 0;
                      0 0 0 0;
                      1 0 0 1; # Row: can I move to 3?
                      0 0 0 0], I(2))

         # CV                       NC
         # P1    P2    ...
         # RL RR RL RR
    B_4 = [0  0  1  0  1  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  1  0  1  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           1  0  0  0  0  0  1  0   1  0  0  0  0  0  1  0; # Move to world with visited cue
           0  1  0  0  0  0  0  1   0  1  0  0  0  0  0  1;
           
           0  0  0  0  0  0  0  0   0  0  1  0  1  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  1  0  1  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0;
           0  0  0  0  0  0  0  0   0  0  0  0  0  0  0  0]

    B = [B_1, B_2, B_3, B_4]

    # Goal prior
    C = softmax(kron(ones(4), [0.0, 0.0, c, -c]))

    # Initial state prior
    #         CV   NC
    D = kron([0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5]) # Start in an NC state

    return (B, C, D)
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

# The primary World defines the Markov blanket between the primary agent and its T-maze environment
function initializePrimaryWorld(B, rs)
    A_s = constructPrimaryA(0.9) # Initialize observation matrix
    function reset(s, a_prime)
       A_s = constructPrimaryA(αs[a_prime]) # Reward probability depends on offer 

       z_0 = zeros(16)
       z_0[8:9] = rs[s]
       z_t_min = z_0
       x_t = A_s*z_0

       return Int64(r'*[2, 3]) # Hidden reward position
    end

    # Set initial reward position (updated by reset)
    r = [0, 1]

    # Initial state
    z_0 = zeros(16)
    z_0[8:9] = r # Start from position 1 in NC state

    # Execute a move to position a_t
    z_t_min = z_0
    function execute(a_t::Int64)
        z_t = B[a_t]*z_t_min # State transition
        x_t = A_s*z_t # Observation probabilities

        z_t_min = z_t # Reset state for next step
    end

    x_t = A_s*z_0
    function observe()
        s = rand(Categorical(x_t))
        o_t = zeros(16)
        o_t[s] = 1.0

        return o_t # One-hot observation
    end

    return (reset, execute, observe)
end
;