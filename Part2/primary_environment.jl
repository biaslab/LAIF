using Random

function constructPrimaryA(α::Float64)
    # Observation model
    #      P1
    #      RL       RR
    #      CV  NC   CV  NC
    A_1 = [0.5 0.5  0.5 0.5;
           0.5 0.5  0.5 0.5;
           0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0]

    #      P2
    #      RL       RR
    #      CV  NC   CV  NC
    A_2 = [0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0;
           α   1.0  1-α 0.0; # Without visiting the cue, all reward goes to the primary agent 
           1-α 0.0  α   1.0]

    #      P3
    #      RL       RR
    #      CV  NC   CV  NC
    A_3 = [0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0;
           1-α 0.0  α   1.0;
           α   1.0  1-α 0.0]

    #      P4
    #      RL       RR
    #      CV  NC   CV  NC
    A_4 = [1.0 1.0  0.0 0.0;
           0.0 0.0  1.0 1.0;
           0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0]

    A = zeros(16, 16)
    A[1:4, 1:4]     = A_1
    A[5:8, 5:8]     = A_2
    A[9:12, 9:12]   = A_3
    A[13:16, 13:16] = A_4

    return A
end

function constructPrimaryBCD(c::Float64)
    # Transition model (with forced move back after reward-arm visit)
    B_1 = kron([1 1 1 1; # Row: can I move to 1?
                0 0 0 0;
                0 0 0 0;
                0 0 0 0], I(4))

    B_2 = kron([0 1 1 0; 
                1 0 0 1; # Row: can I move to 2?
                0 0 0 0;
                0 0 0 0], I(4))

    B_3 = kron([0 1 1 0;
                0 0 0 0;
                1 0 0 1; # Row: can I move to 3?
                0 0 0 0], I(4))

    #      P1           P2 ...
    #      RL    RR        
    #      CV NC CV NC
    B_4 = [0  0  0  0   1  0  0  0   1  0  0  0   0  0  0  0;
           0  0  0  0   0  1  0  0   0  1  0  0   0  0  0  0;
           0  0  0  0   0  0  1  0   0  0  1  0   0  0  0  0;
           0  0  0  0   0  0  0  1   0  0  0  1   0  0  0  0;

           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;

           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;

           1  1  0  0   0  0  0  0   0  0  0  0   1  1  0  0; # Move to World with visited cue
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  1  1   0  0  0  0   0  0  0  0   0  0  1  1; # Move to World with visited cue
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0]

    B = [B_1, B_2, B_3, B_4]

    # Goal prior
    C = softmax(kron(ones(4), [0.0, 0.0, c, -c]))

    # Initial state prior
    #    P1                   P2 ...
    #    RL        RR        
    #    CV   NC   CV   NC   
    D = [0.0, 0.5, 0.0, 0.5,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0]

    return (B, C, D)
end

function generateGoalSequence(S::Int64)
    rs = Vector{Vector}(undef, S)
    for si=1:S
        if rand() > 0.5
            #         P1
            #         RL    RR
            #         CV NC CV NC
            rs[si] = [0, 0, 0, 1]
        else
            rs[si] = [0, 1, 0, 0]
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
       z_0[1:4] = rs[s]
       z_t_min = z_0
       x_t = A_s*z_0

       return Int64(r'*[2, 2, 3, 3]) # Hidden reward position
    end

    # Set initial reward position (updated by reset)
    #    P1
    #    RL    RR
    #    CV NC CV NC
    r = [0, 0, 0, 1]

    # Initial state
    z_0 = zeros(16)
    z_0[1:4] = r # Start from position 1

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