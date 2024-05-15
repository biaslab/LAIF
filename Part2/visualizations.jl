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
                
                ann = (j, i, text(round(dA_x[i,j], digits=1), 10, colour, :center))

                annotate!(px, ann, linecolor=colour)
            end
        end
    end

    plot(p1, p2, p3, p4, layout=grid(1,4,widths=[0.25,0.25,0.25,0.25]), size=(500,220), dpi=300, plot_title=title, plot_titlevspan=0.1)
end

function plotOffers(G_ps, a_ps, o_ps)
    heatmap(hcat(G_ps...), 
            c = cgrad(:grays, rev = true), 
            yticks=(1:L, αs), 
            xlabel="Simulation Trial (s)", 
            ylabel="Offer (α)", 
            dpi=300, 
            size=(600,200),
            left_margin=2Plots.mm,
            bottom_margin=3Plots.mm)

    idx_cv = findall(first.(o_ps).==1.0)
    idx_nc = findall(first.(o_ps).==0.0)
    scatter!(a_ps, color=:white, label=false, markersize=5)
    scatter!(idx_nc, a_ps[idx_nc], label=false, marker=:x, color=:black, markersize=3, markerstrokewidth=3)
end

function plotLearnedGoals(C_0, C, Gs, S)
    G_fix = Matrix{Float64}(undef, 10, S)
    for s=1:S
        G_fix[1,s] = Gs[s][1][1,1]
        G_fix[2,s] = Gs[s][1][1,2]
        G_fix[3,s] = Gs[s][1][1,3]
        G_fix[4,s] = Gs[s][1][1,4]

        G_fix[5,s] = Gs[s][1][2,1]

        G_fix[6,s] = Gs[s][1][3,1]

        G_fix[7,s] = Gs[s][1][4,1]
        G_fix[8,s] = Gs[s][1][4,2]
        G_fix[9,s] = Gs[s][1][4,3]
        G_fix[10,s] = Gs[s][1][4,4]
    end

    yticks = ["P1 CL", "P1 CR", "P1 RW", "P1 NR", 
              "P2 CL", "P2 CR", "P2 RW", "P2 NR",
              "P3 CL", "P3 CR", "P3 RW", "P3 NR",
              "P4 CL", "P4 CR", "P4 RW", "P4 NR"]

    idx1 = [13,14]
    idx2 = [7,8,11,12]

    clim = (0,6)

    p1 = heatmap(hcat([C[1][idx1] .- C_0[1][idx1] for C in Cs]...), 
                c      = cgrad(:grays, rev = true),
                clim   = clim,
                colorbar = false,
                xticks = false,
                #ylabel = "k=1", 
                yflip  = true,
                yticks = (1:2, yticks[idx1]),
                title  = "Learned Goal Statistics")

    p2 = heatmap(hcat([C[2][idx2] .- C_0[2][idx2] for C in Cs]...), 
                c      = cgrad(:grays, rev = true),
                clim   = clim,
                colorbar = false,
                xlabel = "Simulation Trial (s)", 
                #ylabel = "k=2",
                yflip  = true,
                xticks = (1:S, 1:S),
                yticks = (1:4, yticks[idx2]))

    p3 = heatmap(G_fix, 
                c      = cgrad(:grays, rev = true),
                xlabel = "Simulation Trial (s)", 
                #ylabel = "Policy",
                yflip  = true,
                xticks = (1:S, 1:S),
                yticks = (1:10, ["(1,1)", "(1,2)", "(1,3)", "(1,4)", 
                                "(2,1)", 
                                "(3,1)", 
                                "(4,1)", "(4,2)", "(4,3)", "(4,4)"]),
                title  = "Policy GFE [bits]")

    h2 = scatter([0,0], [0,1], 
                zcolor = clim, clims=clim,
                xlims=(1,1.1), xshowaxis=false, yshowaxis=false, 
                label="", c=cgrad(:grays, rev = true), grid=false)

    l = @layout [[a{0.33h}; b{0.66h}] c{0.01w} d{0.95h,0.55w}]

    plot(p1, p2, h2, p3, layout=l, size=(700,280), dpi=300)
end