function averageEnergy(::Type{TransitionMixture},
                       dist_out_in1_switch::Distribution{Multivariate, Contingency},
                       dist_factors::Vararg{Distribution})

    n_factors = length(dist_factors)
    U = 0.0
    for k = 1:n_factors
        U += -tr(dist_out_in1_switch.params[:p][k]'*unsafeLogMean(dist_factors[k]))
    end

    return U
end
