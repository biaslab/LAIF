import ForneyLab: collectSumProductNodeInbounds, collectNaiveVariationalNodeInbounds

using ForneyLab: isClamped, assembleClamp!, unsafeMean, unsafeLogMean
using ForwardDiff: jacobian

using LinearAlgebra: diag


#--------------------------------------
# Variational Rules for DO{Generalized}
#--------------------------------------

# Message towards state
@naiveVariationalRule(:node_type     => DiscreteObservation{Generalized},
                      :outbound_type => Message{Categorical},
                      :inbound_types => (Distribution, Nothing, Distribution, Distribution),
                      :name          => VBDiscreteObservationGeneralizedS)

# Unobserved
function ruleVBDiscreteObservationGeneralizedS(
    ::Distribution{Univariate, Categorical}, # Unconstrained observation (not used)
    msg_s::Message{Categorical, Univariate},
    marg_s::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    marg_c::Distribution{Multivariate};
    n_iterations=20)

    d = msg_s.dist.params[:p]
    s_0 = unsafeMean(marg_s)
    A = unsafeMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    # Root-finding problem for marginal statistics
    g(s) = s - softmax(diag(A'*safelog.(A)) + A'*log_c - A'*safelog.(A*s) + safelog.(d))

    s_k = deepcopy(s_0)
    for k=1:n_iterations
        s_k = s_k - inv(jacobian(g, s_k))*g(s_k) # Newton step for multivariate root finding
    end

    # Compute unnormalized outbound message statistics
    rho = s_k./(d .+ 1e-6)

    Message(Univariate, Categorical, p=rho./sum(rho))
end

# Observed
function ruleVBDiscreteObservationGeneralizedS(
    marg_y::Distribution{Multivariate, PointMass}, # Constrained observation
    ::Message, # State message not used
    ::Any,
    marg_A::Distribution{MatrixVariate},
    ::Any; # Goal marginal not used
    n_iterations=20) # Iterations not used

    y_hat = unsafeMean(marg_y)
    log_A = unsafeLogMean(marg_A)

    Message(Univariate, Categorical, p=softmax(log_A'*y_hat))
end

# Message towards goal
@naiveVariationalRule(:node_type     => DiscreteObservation{Generalized},
                      :outbound_type => Message{Dirichlet},
                      :inbound_types => (Distribution, Distribution, Distribution, Nothing),
                      :name          => VBDiscreteObservationGeneralizedC)

# Unobserved
function ruleVBDiscreteObservationGeneralizedC(
    ::Distribution{Univariate, Categorical}, # Unconstrained observation (not used)
    marg_s::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    marg_c::Any)

    s = unsafeMean(marg_s)
    A = unsafeMean(marg_A)

    Message(Multivariate, Dirichlet, a=A*s .+ 1)
end

# Observed
function ruleVBDiscreteObservationGeneralizedC(
    marg_y::Distribution{Multivariate, PointMass}, # Constrained observation
    ::Any, # State marginal not used
    ::Any, # Parameter marginal not used
    ::Any)

    y_hat = unsafeMean(marg_y)

    Message(Multivariate, Dirichlet, a=y_hat .+ 1)
end

# Message towards parameter
@naiveVariationalRule(:node_type     => DiscreteObservation{Generalized},
                      :outbound_type => Message{Dirichlet}, # Returns Function message in unconstrained case
                      :inbound_types => (Distribution, Distribution, Nothing, Distribution),
                      :name          => VBDiscreteObservationGeneralizedA)

# Unobserved
function ruleVBDiscreteObservationGeneralizedA(
    ::Distribution{Univariate, Categorical}, # Unconstrained observation (not used)
    marg_s::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    marg_c::Distribution{Multivariate})

    s = unsafeMean(marg_s)
    A = unsafeMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    log_mu_A(Z) = s'*diag(Z'*safelog.(Z)) + (Z*s)'*log_c - (Z*s)'*safelog.(A*s)

    Message(MatrixVariate, Function, log_pdf=log_mu_A)
end

# Observed
function ruleVBDiscreteObservationGeneralizedA(
    marg_y::Distribution{Multivariate, PointMass}, # Constrained observation
    marg_s::Distribution{Univariate},
    ::Any,
    ::Any) # Goal marginal not used

    y_hat = unsafeMean(marg_y)
    s = unsafeMean(marg_s)

    Message(MatrixVariate, Dirichlet, a=y_hat*s' .+ 1) # Return Dirichlet message instead of Function
end

# Message towards observation
@naiveVariationalRule(:node_type     => DiscreteObservation{Generalized},
                      :outbound_type => Message{Categorical},
                      :inbound_types => (Nothing, Distribution, Distribution, Distribution),
                      :name          => VBDiscreteObservationGeneralizedY)

# Unobserved
function ruleVBDiscreteObservationGeneralizedY(
    ::Distribution{Univariate, Categorical},
    marg_s::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    ::Any) # Goal marginal not used

    s = unsafeMean(marg_s)
    A = unsafeMean(marg_A)

    Message(Univariate, Categorical, p=A*s)
end

# Observed
function ruleVBDiscreteObservationGeneralizedY(
    marg_y::Distribution{Multivariate, PointMass},
    ::Any, # State marginal not used
    ::Any, # Parameter marginal not used
    ::Any) # Goal marginal not used

    y_hat = unsafeMean(marg_y)

    Message(Multivariate, PointMass, m=y_hat) # Clamped marginal remains clamped
end

# Custom inbounds collector
function collectNaiveVariationalNodeInbounds(node::DiscreteObservation{Generalized}, entry::ScheduleEntry)
    algo = currentInferenceAlgorithm()
    interface_to_schedule_entry = algo.interface_to_schedule_entry
    target_to_marginal_entry = algo.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface === node.interfaces[2]
            # Outbound message for s: collect inbound message and marginal
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif node_interface === node.interfaces[1]
            # Marginal for y is always included (for overloading)
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif node_interface === node.interfaces[3]
            # Marginal for A is always included (for dependency)
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif node_interface === entry.interface
            # Otherwise do not collect inbounds for remaining outbound messages
            push!(inbounds, nothing)
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Distribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    # Push custom arguments if defined
    if (node.n_iterations !== nothing)
        push!(inbounds, Dict{Symbol, Any}(:n_iterations => node.n_iterations,
                                          :keyword      => true))
    end

    return inbounds
end
