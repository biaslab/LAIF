import ForneyLab: collectSumProductNodeInbounds, collectNaiveVariationalNodeInbounds

using ForneyLab: isClamped, assembleClamp!, unsafeMean, unsafeLogMean
using ForwardDiff: jacobian

using LinearAlgebra: diag

export ruleVBDiscreteObservationOut, ruleVBDiscreteObservationOut, softmax


#-----------------
# Sum-Product Rule
#-----------------

@sumProductRule(:node_type     => DiscreteObservation,
                :outbound_type => Message{Categorical},
                :inbound_types => (Nothing, Message{PointMass}, Message{PointMass}),
                :name          => SPDiscreteObservationOutDPP)

function ruleSPDiscreteObservationOutDPP(
    msg_out::Message{Categorical, Univariate},
    marg_out::Distribution{Univariate, Categorical},
    msg_A::Message{PointMass, MatrixVariate},
    msg_c::Message{PointMass, Multivariate};
    n_iterations=20)

    d = msg_out.dist.params[:p]
    s_0 = marg_out.params[:p]
    A = msg_A.dist.params[:m]
    log_c = unsafeLogMean(msg_c.dist)

    rho = msgDiscreteObservationOut(d, s_0, A, log_c, n_iterations)

    Message(Univariate, Categorical, p=rho)
end

function collectSumProductNodeInbounds(node::DiscreteObservation, entry::ScheduleEntry)
    algo = currentInferenceAlgorithm()
    interface_to_schedule_entry = algo.interface_to_schedule_entry
    target_to_marginal_entry = algo.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Collect inbound message
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
            # Collect inbound marginal
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif isClamped(inbound_interface)
            # Hard-code outbound message of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Message))
        else
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    # Push custom arguments if defined
    if (node.n_iterations !== nothing)
        push!(inbounds, Dict{Symbol, Any}(:n_iterations => node.n_iterations,
                                          :keyword      => true))
    end

    return inbounds
end


#------------------
# Variational Rules
#------------------

@naiveVariationalRule(:node_type     => DiscreteObservation,
                      :outbound_type => Message{Categorical},
                      :inbound_types => (Nothing, Distribution, Distribution),
                      :name          => VBDiscreteObservationOut)

function ruleVBDiscreteObservationOut(
    msg_out::Message{Categorical, Univariate},
    marg_out::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    marg_c::Distribution{Multivariate};
    n_iterations=20)

    d = msg_out.dist.params[:p]
    s_0 = unsafeMean(marg_out)
    A = unsafeMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    rho = msgDiscreteObservationOut(d, s_0, A, log_c, n_iterations)

    Message(Univariate, Categorical, p=rho)
end

@naiveVariationalRule(:node_type     => DiscreteObservation,
                      :outbound_type => Message{Dirichlet},
                      :inbound_types => (Distribution, Distribution, Nothing),
                      :name          => VBDiscreteObservationC)

function ruleVBDiscreteObservationC(
    marg_out::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    marg_c::Any)

    s = unsafeMean(marg_out)
    A = unsafeMean(marg_A)

    Message(Multivariate, Dirichlet, a=A*s .+ 1)
end

@naiveVariationalRule(:node_type     => DiscreteObservation,
                      :outbound_type => Message{Function},
                      :inbound_types => (Distribution, Nothing, Distribution),
                      :name          => VBDiscreteObservationA)

function ruleVBDiscreteObservationA(
    marg_out::Distribution{Univariate},
    marg_A::Distribution{MatrixVariate},
    marg_c::Distribution{Multivariate})

    s = unsafeMean(marg_out)
    A = unsafeMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    log_mu_A(Z) = s'*diag(Z'*safelog.(Z)) + (Z*s)'*log_c - (Z*s)'*safelog.(A*s)

    Message(MatrixVariate, Function, log_pdf=log_mu_A)
end

function collectNaiveVariationalNodeInbounds(node::DiscreteObservation, entry::ScheduleEntry)
    algo = currentInferenceAlgorithm()
    interface_to_schedule_entry = algo.interface_to_schedule_entry
    target_to_marginal_entry = algo.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface === node.interfaces[1]
            # For outbound message on out interface, collect inbound message and marginal
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif node_interface === entry.interface === node.interfaces[2]
            # For outbound message on A interface, collect inbound marginal
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif node_interface === entry.interface
            # Do not collect inbounds for remaining outbound messages
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


#---------------
# Shared updates
#---------------

function msgDiscreteObservationOut(d::Vector, s_0::Vector, A::Matrix, log_c::Vector, n_iterations::Int64)
    # Root-finding problem for marginal statistics
    g(s) = s - softmax(diag(A'*safelog.(A)) + A'*log_c - A'*safelog.(A*s) + safelog.(d))

    s_k = deepcopy(s_0)
    for k=1:n_iterations
        s_k = s_k - inv(jacobian(g, s_k))*g(s_k) # Newton step for multivariate root finding
    end

    # Compute outbound message statistics
    rho = s_k./(d .+ 1e-6)
    return rho./sum(rho)
end
