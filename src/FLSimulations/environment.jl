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

    # Transition model (original)
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

     # Transition model (adjusted)
#    B_1 = kron([1 1 1 1; # Row: can I move to 1?
#                0 0 0 0;
#                0 0 0 0;
#                0 0 0 0], I(2))
#
#    B_2 = kron([0 1 1 0;
#                1 0 0 1; # Row: can I move to 2?
#                0 0 0 0;
#                0 0 0 0], I(2))
#
#    B_3 = kron([0 1 1 0;
#                0 0 0 0;
#                1 0 0 1; # Row: can I move to 3?
#                0 0 0 0], I(2))
#
#    B_4 = kron([0 1 1 0;
#                0 0 0 0;
#                0 0 0 0;
#                1 0 0 1], I(2)) # Row: can I move to 4?

    B = [B_1, B_2, B_3, B_4]

    # Goal prior
    C = softmax(kron(ones(4), [0.0, 0.0, c, -c]))

    # Initial state prior
    # Note: in the epistemic value paper (Friston, 2015) there is a softmax over D.
    # However, from the context as described in the paper this appears to be a notational error.
    D = kron([1.0, 0.0, 0.0, 0.0], [0.5, 0.5])

    return (A, B, C, D)
end
;
