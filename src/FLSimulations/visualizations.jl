using LaTeXStrings
using SparseArrays

function plotFreeEnergies(Gt::Vector, at::Vector, ot::Vector, r::Int64)
    min1 = floor(minimum(skipmissing(Gt[1])))
    max1 = ceil(maximum(skipmissing(Gt[1])))
    min2 = floor(minimum(skipmissing(Gt[2])))
    max2 = ceil(maximum(skipmissing(Gt[2])))

    p1 = plotG1(Gt[1], at, ot, r, clim=(min1,max1+0.5))
    p2 = plotG2(Gt[2], at, clim=(min2,max2+0.5))

    plot(p1, p2, layout=grid(2,1,heights=[0.8,0.2]), size=(300, 400))
end

function plotG1(F::Matrix, at::Vector, ot::Vector, r::Int64; 
                dpi=100, clim=(4.0,8.0), title="", highlight=minimum)
    p = heatmap(F,
            dpi=dpi,
            color=:grays, 
            aspect_ratio=:equal,
            colorbar=false,
            xlim=(0.5,4.5), 
            ylim=(0.5,4.5), 
            # ylabel=L"\mathrm{First\,Policy\,} (\pi_1)",
            title=title,
            clim=clim,
            xticks=false,
            xtickfontsize=12,
            ytickfontsize=12,
            xguidefontsize=14,
            yguidefontsize=14)

    F_round = round.(F, digits=1)
    if highlight !== nothing
        extremum = highlight(skipmissing(F_round))
    else
        extremum = NaN
    end

    for i=1:4
        for j=1:4
            ismissing(F[i,j]) && continue

            # Annotate number
            if F[i,j] >= clim[2]-0.3
                colour = :black
            else
                colour = :white
            end
            
            if extremum == F_round[i,j]
                ann = (j, i, text("$(F_round[i,j])*", 15, :red, :center)) # Annotate extremum
            else
                ann = (j, i, text(F_round[i,j], 15, colour, :center))
            end
            annotate!(ann, linecolor=colour)
        end
    end

    mask = kron(ones(4), [0.0, 0.0, 1.0, 0.0])
    rewarded = (dot.([mask], ot).== 1.0) # Obtained reward
    rewarded_dict = Dict{Bool, String}(true  => "●",
                                      false => "○")
    txt = join([rewarded_dict[r_t] for r_t in rewarded], " ")
    ann = (3, 2.5, text(txt, 18, :gold, :center))
    annotate!(ann, linecolor=:gold)

    ann = (3, 2.45, text(join(at, "   "), 13, :black, :center)) # Executed actions
    annotate!(ann, linecolor=:black)

    ct = [Int(ot[t][13:14]'*[2, 3]) for t=1:2] # Received cues
    cue_dict = Dict{Int, String}(0 => " ",
                                 2 => "←",
                                 3 => "→")
    ann = (3, 2.1, text(join([cue_dict[ct[t]] for t=1:2], "  "), 18, :gold, :center))
    annotate!(ann, linecolor=:gold)

    return p
end

function plotG2(F::Matrix, at::Vector; dpi=100, clim=(4.0,8.0), title="", highlight=minimum)
    F = reshape(F[at[1], :], 1, 4)
    p = heatmap(F,
            dpi=dpi,
            color=:grays, 
            aspect_ratio=:equal,
            colorbar=false,
            xlim=(0.5,4.5), 
            ylim=(0.5,1.5), 
            # xlabel=L"\mathrm{Second\,Policy\,} (\pi_2)",
            title=title,
            clim=clim,
            yticks=([1.0], [at[1]]),
            xtickfontsize=12,
            ytickfontsize=12,
            xguidefontsize=14,
            yguidefontsize=14)

    F_round = round.(F, digits=1)
    if highlight !== nothing
        extremum = highlight(skipmissing(F_round))
    else
        extremum = NaN
    end

    for j=1:4
        ismissing(F[j]) && continue

        # Annotate number
        if F[j] >= clim[2]-0.3
            colour = :black
        else
            colour = :white
        end
        
        if extremum == F_round[j]
            ann = (j, 1, text("$(F_round[j])*", 15, :red, :center)) # Annotate extremum
        else
            ann = (j, 1, text(F_round[j], 15, colour, :center))
        end
        annotate!(ann, linecolor=colour)
    end

    return p
end

function plotFreeEnergyMinimum(Gts)
    S = length(Gts)
    
    # Plot free energies over simulations
    G1_mins = [minimum(skipmissing(Gts[s][1])) for s=1:S]
    G2_mins = [minimum(skipmissing(Gts[s][2])) for s=1:S]
    G3s = [minimum(skipmissing(Gts[s][3])) for s=1:S]

    plot(1:S, G1_mins, xlabel="Simulation (s)", ylabel="Free Energy Minimum [bits]", label="t=1", lw=2)
    plot!(1:S, G2_mins, label="t=2", lw=2)
    plot!(1:S, G3s, label="t=3", lw=2)
end

function plotDecomposition(polts, riskts, ambts, novts)
    S = length(polts)
    
    # Extract risks for t=1
    risks = [riskts[s][1][polts[s][1]] for s=1:S]
    ambs = [ambts[s][1][polts[s][1]] for s=1:S]
    novs = [novts[s][1][polts[s][1]] for s=1:S]

    plot(1:S, risks, xlabel="Simulation (s)", ylabel="Value [bits]", label="Risk", lw=2, title="Value decomposition for best policy at t=1") #, yaxis=:log)
    plot!(1:S, ambs, label="Ambiguity", lw=2)
    plot!(1:S, novs, label="Novelty", lw=2)
end

function plotObservationStatistics(A::Matrix, A_0::Matrix)
    # Inspect difference in observation statistics
    dA = sparse(round.(A - A_0, digits=1)) 
    dA_1 = dA[1:4, 1:2]
    dA_2 = dA[5:8, 3:4]
    dA_3 = dA[9:12, 5:6]
    dA_4 = dA[13:16, 7:8]

    return [dA_1 dA_2 dA_3 dA_4]
end