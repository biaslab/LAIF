using LaTeXStrings
using SparseArrays

function plotFreeEnergies(Gt::Vector, at::Vector, ot::Vector, r::Vector; title=title)
    min1 = floor(minimum(skipmissing(Gt[1])))
    max1 = ceil(maximum(skipmissing(Gt[1])))
    min2 = floor(minimum(skipmissing(Gt[2])))
    max2 = ceil(maximum(skipmissing(Gt[2])))

    p1 = plotG1(Gt[1], at, ot, r, clim=(min1,max1+0.5), title=title)
    p2 = plotG2(Gt[2], at, clim=(min2,max2+0.5))

    plot(p1, p2, layout=grid(2,1,heights=[0.8,0.2]), size=(300, 400), dpi=300)
end

function plotG1(F::Matrix, at::Vector, ot::Vector, r::Vector; 
                clim=(4.0,8.0), title="", highlight=minimum)
    ticks = ([1,2,3,4], ["O","L","R","C"])
    p = heatmap(F,
            dpi=300,
            color=:grays, 
            aspect_ratio=:equal,
            colorbar=false,
            xlim=(0.5,4.5), 
            ylim=(0.5,4.5), 
            title=title,
            clim=clim,
            xticks=false,
            yticks=ticks,
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

    obs_mask = kron(ones(Int64, 4), [1, 2, 3, 4])
    obs_dict = Dict{Int, String}(1 => "CL",
                                 2 => "CR",
                                 3 => "RW",
                                 4 => "NR")
    sta_mask = kron([1, 2, 3, 4], ones(Int64, 4))
    sta_dict = Dict{Int, String}(1 => "O",
                                 2 => "L",
                                 3 => "R",
                                 4 => "C")

    obs = dot.([obs_mask], ot) # Observation
    sta = dot.([sta_mask], ot) # State

    txt = join([sta_dict[s_t] for s_t in sta], "   ")
    ann = (3, 2.8, text(txt, 18, :black, :center))
    annotate!(ann, linecolor=:black)

    txt = join([obs_dict[o_t] for o_t in obs], " ")
    ann = (3, 2.2, text(txt, 18, :black, :center))
    annotate!(ann, linecolor=:black)

    return p
end

function plotG2(F::Matrix, at::Vector; clim=(4.0,8.0), title="", highlight=minimum)
    ticks = ([1,2,3,4], ["O","L","R","C"])
    F = reshape(F[at[1], :], 1, 4)
    p = heatmap(F,
            dpi=300,
            color=:grays, 
            aspect_ratio=:equal,
            colorbar=false,
            xlim=(0.5,4.5), 
            ylim=(0.5,1.5), 
            title=title,
            clim=clim,
            xticks=ticks,
            yticks=([1.0], ticks[2][at[1]]),
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

function plotFreeEnergyMinimum(Gs, os; args...)
    S = length(Gs)
    
    # Plot free energies over simulations
    G1_mins = [minimum(skipmissing(Gs[s][1])) for s=1:S]
    G2_mins = [minimum(skipmissing(Gs[s][2])) for s=1:S]
    G3s = [minimum(skipmissing(Gs[s][3])) for s=1:S]

    empty_ticks = ([0,25,50,75,100],["","","","",""])
    p1 = plot(1:S, G1_mins, xticks=empty_ticks, ylabel="Free Energy Minimum [bits]", label="t=1", lw=2, linestyle=:dashdot; args...)
    plot!(p1, 1:S, G2_mins, label="t=2", lw=2, linestyle=:dash)
    plot!(p1, 1:S, G3s, label="t=3", lw=2)

    wins = extractWins(os)
    p2 = scatter(1:S, 1 .- wins, xlabel="Simulation Trial (s)", yticks=([0, 1], ["win", "loss"]), color=:black, legend=false, ylim=(-0.1, 1.1), markersize=2.5) # Plot non-reward
    
    plot(p1, p2, layout=grid(2,1,heights=[0.8,0.2]), dpi=300)
end

function extractWins(os)
    win_mask = kron(ones(Int64, 4), [0,0,1,0])
    wins = Vector{Float64}(undef, S)
    for s=1:S
        win_1 = win_mask'*os[s][1]
        win_2 = win_mask'*os[s][2]
        wins[s] = win_1 + win_2
    end

    return wins
end

function plotObservationStatistics(A::Matrix, A_0::Matrix; title="")
    # Inspect difference in observation statistics
    # dA = sparse(round.(A - A_0, digits=1))
    dA = Matrix{Union{Missing,Float64}}(A - A_0)
    zs = (dA .< 0.01)
    dA[zs] .= missing
    dA_1 = dA[1:4, 1:2]
    dA_2 = dA[5:8, 3:4]
    dA_3 = dA[9:12, 5:6]
    dA_4 = dA[13:16, 7:8]
    cmax = maximum(skipmissing(dA))

    yticks = ([1, 2, 3, 4], ["CL", "CR", "RW", "NR"])
    empty = ([1, 2, 3, 4], ["", "", "", ""])
    xticks = ([1, 2], ["RL", "RR"])
    cg = cgrad(:grays, rev = true)
    p1 = heatmap(dA_1, title=L"O", yflip=true, c=cg, colorbar=false, clim=(0, 55), color=:grays, xticks=xticks, yticks=yticks)
    p2 = heatmap(dA_2, title=L"L", yflip=true, c=cg, colorbar=false, clim=(0, 55), color=:grays, xticks=xticks, yticks=empty)
    p3 = heatmap(dA_3, title=L"R", yflip=true, c=cg, colorbar=false, clim=(0, 55), color=:grays, xticks=xticks, yticks=empty)
    p4 = heatmap(dA_4, title=L"C", yflip=true, c=cg, colorbar=false, clim=(0, 55), xticks=xticks, yticks=empty)

    for (px, dA_x) in [(p1,dA_1), (p2,dA_2), (p3,dA_3), (p4,dA_4)]
        for i=1:4
            for j=1:2
                ismissing(dA_x[i,j]) && continue

                # Annotate number
                if dA_x[i,j] <= 30
                    colour = :black
                else
                    colour = :white
                end
                
                ann = (j, i, text(Int64(round(dA_x[i,j], digits=0)), 10, colour, :center))

                annotate!(px, ann, linecolor=colour)
            end
        end
    end

    plot(p1, p2, p3, p4, layout=grid(1,4,widths=[0.25,0.25,0.25,0.25]), size=(500,220), dpi=300, plot_title=title, plot_titlevspan=0.1)
end