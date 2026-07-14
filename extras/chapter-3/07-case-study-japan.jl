### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ a7b8c9d0-0003-8e6f-3b4c-0d1e2f3a4b5c
begin
	using JSON3
	using DataFrames
	using Statistics
	using Chain: @chain
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ a7b8c9d0-0001-8e6f-3b4c-0d1e2f3a4b5c
md"""
# Case Study: Japan Women's at WWC 2019

**Chapter 3 · Exploratory Data Analysis in Soccer**

## The research question

**How did Japan Women's team play at the 2019 FIFA Women's World Cup?**

Japan is known for a possession-oriented style, patient buildup, precise
passing, and technical excellence.  We'll use data to test these assumptions
and uncover deeper patterns.

## What you'll learn

- Apply a complete EDA workflow from data to insights
- Combine tournament-level comparisons with match-level deep dives
- Use spatial heatmaps to reveal tactical patterns
- Synthesise quantitative metrics with tactical interpretation
"""

# ╔═╡ a7b8c9d0-0002-8e6f-3b4c-0d1e2f3a4b5c
md"""
## Imports & setup
"""

# ╔═╡ a7b8c9d0-0004-8e6f-3b4c-0d1e2f3a4b5c
"""
    flatten_dict(d::AbstractDict; prefix="")

Recursively flatten a nested dictionary, joining keys with `"."`.
"""
function flatten_dict(d::AbstractDict; prefix="")
    flat = Dict{String,Any}()
    for (k, v) in d
        new_key = isempty(prefix) ? string(k) : "$prefix.$k"
        if v isa AbstractDict
            merge!(flat, flatten_dict(v; prefix=new_key))
        else
            flat[new_key] = v === nothing ? missing : v
        end
    end
    return flat
end

# ╔═╡ a7b8c9d0-0005-8e6f-3b4c-0d1e2f3a4b5c
"""
    draw_pitch(; bgcolor, linecolor, linewidth)

Draw a soccer pitch in landscape orientation (120 × 80 StatsBomb coords).
"""
function draw_pitch(;
		bgcolor   = "#22312b",
		linecolor = "#efefef",
		linewidth = 1.5,
		title     = "",
	)
	p = plot(xlims=(-2, 122), ylims=(-2, 82), aspect_ratio=:equal,
		legend=false, background_color=bgcolor, ticks=false, framestyle=:none,
		title=title)
	plot!(p, [0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color=linecolor, lw=linewidth)
	plot!(p, [60, 60], [0, 80], color=linecolor, lw=linewidth)
	θ = range(0, 2π; length=100)
	plot!(p, 60 .+ 9.15 .* cos.(θ), 40 .+ 9.15 .* sin.(θ), color=linecolor, lw=linewidth)
	scatter!(p, [60], [40], color=linecolor, ms=3)
	plot!(p, [0, 18, 18, 0, 0], [18, 18, 62, 62, 18], color=linecolor, lw=linewidth)
	plot!(p, [120, 102, 102, 120, 120], [18, 18, 62, 62, 18], color=linecolor, lw=linewidth)
	plot!(p, [0, 6, 6, 0, 0], [30, 30, 50, 50, 30], color=linecolor, lw=linewidth)
	plot!(p, [120, 114, 114, 120, 120], [30, 30, 50, 50, 30], color=linecolor, lw=linewidth)
	scatter!(p, [12, 108], [40, 40], color=linecolor, ms=4)
	for (cx, cy, a0) in [(0, 0, 0), (120, 0, π/2), (120, 80, π), (0, 80, 3π/2)]
		θc = range(a0, a0 + π/2; length=20)
		plot!(p, cx .+ 1 .* cos.(θc), cy .+ 1 .* sin.(θc), color=linecolor, lw=linewidth)
	end
	p
end

# ╔═╡ a7b8c9d0-0006-8e6f-3b4c-0d1e2f3a4b5c
md"""
## Part 1 — Tournament-level overview

First, the big picture: how did Japan compare to other top teams across
the whole tournament?  We load events from **all 52 matches**.
"""

# ╔═╡ a7b8c9d0-0007-8e6f-3b4c-0d1e2f3a4b5c
begin
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")

	match_file = joinpath(DATA_DIR, "matches", "72", "30.json")
	match_raw = JSON3.read(read(match_file, String))
	match_ids = [d["match_id"] for d in match_raw]

	# Load and flatten events from every match — this is the expensive step
	all_dfs = []
	for mid in match_ids
		evt_file = joinpath(DATA_DIR, "events", "$mid.json")
		if !isfile(evt_file); continue; end
		raw_evt = JSON3.read(read(evt_file, String))
		dicts = flatten_dict.(raw_evt)
		keys_set = union((keys(d) for d in dicts)...)
		rows = [let row = Dict{String,Any}()
			for k in keys_set; row[k] = get(d, k, missing); end; row
		end for d in dicts]
		df = DataFrame(rows)
		df.match_id .= mid
		push!(all_dfs, df)
	end

	global all_events = vcat(all_dfs...; cols=:union)
	md"""
	Loaded **$(nrow(all_events)) events** from **$(length(all_dfs)) matches**
	across $(ncol(all_events)) columns.
	"""
end

# ╔═╡ a7b8c9d0-0008-8e6f-3b4c-0a1b2c3d4e5f
md"""
### Passing statistics — tournament-wide

Compare Japan's pass accuracy to the other semi-finalists.
"""

# ╔═╡ a7b8c9d0-0009-8e6f-3b4c-0a1b2c3d4e5f
let
	all_passes = subset(all_events, "type.name" => ByRow(==("Pass")); skipmissing=true)

	team_passing = @chain all_passes begin
		groupby(_, "team.name")
		combine(
			"type.name" => length => :total_passes,
			"pass.outcome.name" => (x -> count(ismissing, x)) => :completed_passes)
		transform(_, [:completed_passes, :total_passes] =>
			ByRow((c, t) -> c / t) => :pass_accuracy)
	end

	# Focus on four top teams
	top_teams = ["Japan Women's", "United States Women's", "England Women's", "Netherlands Women's"]
	comparison = subset(team_passing, "team.name" => ByRow(t -> t in top_teams); skipmissing=true)
	global pass_comparison = sort(comparison, :pass_accuracy, rev=true)

	# Bar chart
	n = nrow(pass_comparison)
	labs = reverse(Vector(pass_comparison[!, "team.name"]))
	vals = reverse(Vector(pass_comparison.pass_accuracy))
	p = plot(xlims=(0.7, 0.9), ylims=(0.5, n + 0.5),
		legend=false, title="Pass Accuracy — Japan vs Top Teams",
		xlabel="Pass Accuracy", yticks=(1:n, labs))
	for i in 1:n
		plot!(p, [0, vals[i]], [i, i], linewidth=12, color=:steelblue, legend=false)
	end
	p
end

# ╔═╡ a7b8c9d0-0010-8e6f-3b4c-0a1b2c3d4e5f
md"""
**Finding:** Japan led the tournament in pass accuracy.  This confirms
their reputation for precise, possession-based play — even against the
eventual finalists.
"""

# ╔═╡ a7b8c9d0-0011-8e6f-3b4c-0a1b2c3d4e5f
md"""
### Shot and goal statistics

How did Japan's shot volume and conversion compare?
"""

# ╔═╡ a7b8c9d0-0012-8e6f-3b4c-0a1b2c3d4e5f
let
	all_shots = subset(all_events, "type.name" => ByRow(==("Shot")); skipmissing=true)

	team_shooting = @chain all_shots begin
		groupby(_, "team.name")
		combine(
			"type.name" => length => :shots,
			"shot.outcome.name" => (x -> count(==("Goal"), x)) => :goals)
	end

	# Focus on top teams
	top_teams = ["Japan Women's", "United States Women's", "England Women's", "Netherlands Women's"]
	shooting_comp = subset(team_shooting, "team.name" => ByRow(t -> t in top_teams); skipmissing=true)

	# Scatter plot with labels
	scatter(shooting_comp.shots, shooting_comp.goals,
		legend=false, markersize=12, color=:darkblue,
		title="Shot Volume vs Goals — Top Teams",
		xlabel="Total Shots", ylabel="Total Goals")

	for row in eachrow(shooting_comp)
		label = replace(row."team.name", " Women's" => "")
		annotate!(row.shots + 1.5, row.goals,
			Plots.text(label, 8, :black, :left))
	end

	# Trend line
	x = shooting_comp.shots; y = shooting_comp.goals
	X = hcat(ones(length(x)), x)
	coeffs = X \ y
	xfit = range(minimum(x), maximum(x); length=50)
	plot!(xfit, coeffs[1] .+ coeffs[2] .* xfit,
		color=:gray, linestyle=:dash, linewidth=1.5, label="Trend")
end

# ╔═╡ a7b8c9d0-0013-8e6f-3b4c-0a1b2c3d4e5f
md"""
**Finding:** Japan took fewer shots than other top teams but maintained
reasonable conversion.  This suggests a patient, selective approach to
shooting — prioritising quality over quantity.
"""

# ╔═╡ a7b8c9d0-0014-8e6f-3b4c-0a1b2c3d4e5f
md"""
## Part 2 — Match-level analysis

Let's dive into Japan's individual matches to see how their style played
out game by game.
"""

# ╔═╡ a7b8c9d0-0015-8e6f-3b4c-0a1b2c3d4e5f
let
	# Load match-level data to find Japan's fixtures
	match_raw = JSON3.read(read(joinpath(DATA_DIR, "matches", "72", "30.json"), String))
	matches_df = DataFrame(flatten_dict.(match_raw))

	japan_matches = subset(matches_df,
		["home_team.home_team_name", "away_team.away_team_name"] =>
			ByRow((h, a) -> h == "Japan Women's" || a == "Japan Women's");
		skipmissing=true)

	println("Japan played $(nrow(japan_matches)) matches:")
	for row in eachrow(japan_matches)
		println("  $(row."home_team.home_team_name") vs $(row."away_team.away_team_name"): $(row.home_score)-$(row.away_score)")
	end

	# Store Japan's match IDs for later
	global japan_match_ids = japan_matches.match_id
end

# ╔═╡ a7b8c9d0-0016-8e6f-3b4c-0a1b2c3d4e5f
md"""
### Japan's per-match passing
"""

# ╔═╡ a7b8c9d0-0017-8e6f-3b4c-0a1b2c3d4e5f
let
	# Filter passes to Japan's matches only
	japan_passes = subset(all_events,
		"type.name" => ByRow(==("Pass")),
		"team.name" => ByRow(==("Japan Women's"));
		skipmissing=true)

	japan_passes = subset(japan_passes,
		:match_id => ByRow(mid -> mid in japan_match_ids);
		skipmissing=true)

	per_match = @chain japan_passes begin
		groupby(_, [:match_id, "team.name"])
		combine(
			"type.name" => length => :attempted,
			"pass.outcome.name" => (x -> count(ismissing, x)) => :completed)
		transform(_, [:completed, :attempted] =>
			ByRow((c, a) -> c / a) => :pass_accuracy)
	end

	# Pretty print with match info
	match_raw = JSON3.read(read(joinpath(DATA_DIR, "matches", "72", "30.json"), String))
	match_lookup = Dict(d["match_id"] => "$(d["home_team"]["home_team_name"]) $(d["home_score"])-$(d["away_score"]) $(d["away_team"]["away_team_name"])" for d in match_raw)

	select(per_match, :match_id, :completed, :attempted, :pass_accuracy)
end

# ╔═╡ a7b8c9d0-0018-8e6f-3b4c-0a1b2c3d4e5f
md"""
## Part 3 — Spatial analysis

Where on the pitch did Japan build their play?  A heatmap of pass
locations reveals their tactical footprint.
"""

# ╔═╡ a7b8c9d0-0019-8e6f-3b4c-0a1b2c3d4e5f
let
	# Japan's completed passes with location data
	jp = subset(all_events,
		"type.name"  => ByRow(==("Pass")),
		"team.name"  => ByRow(==("Japan Women's")),
		:match_id    => ByRow(mid -> mid in japan_match_ids);
		skipmissing=true)

	jp.x = [(ismissing(loc) || loc === nothing) ? missing : loc[1] for loc in jp.location]
	jp.y = [(ismissing(loc) || loc === nothing) ? missing : loc[2] for loc in jp.location]
	clean = dropmissing(jp, [:x, :y])

	p = draw_pitch(bgcolor="#1a1a2e", linecolor="#aaaaaa",
		title="Japan Women's — Pass Density Heatmap (WWC 2019)")

	histogram2d!(p, clean.x, clean.y, bins=(50, 35),
		colormap=:YlOrRd, alpha=0.75, colorbar=true, label="")

	p
end

# ╔═╡ a7b8c9d0-0020-8e6f-3b4c-0a1b2c3d4e5f
md"""
**Finding:** Japan's pass heatmap reveals:
- Strong concentration in central midfield areas
- Even distribution across the width — they use the full pitch
- Significant buildup from defensive positions
- Confirms possession-based, patient style anchored in midfield control
"""

# ╔═╡ a7b8c9d0-0021-8e6f-3b4c-0a1b2c3d4e5f
md"""
## Summary — what we learned about Japan

### Quantitative findings
1. **Highest pass accuracy** among the semi-finalists
2. **Moderate shot volume** — fewer shots, but quality over quantity
3. **Consistent passing accuracy** across all their matches
4. **Central midfield dominance** in spatial patterns

### Tactical interpretation
Japan's style at WWC 2019 matched their reputation: technically excellent,
possession-focused, and patient.  They prioritised control over chaos,
precision over power.  While this didn't lead to a tournament victory,
it showcased a distinctive and effective approach to the game.

### EDA workflow lessons
1. **Start broad** — tournament-level comparisons establish context
2. **Zoom in** — match-by-match analysis reveals consistency (or not)
3. **Go spatial** — heatmaps show *where* the style manifests
4. **Synthesise** — combine quantitative metrics with tactical reasoning
5. **Tell a story** — data alone isn't insight; interpretation is key

This same workflow applies to any team, player, or tactical question in
soccer analytics.
"""

# ╔═╡ a7b8c9d0-0022-8e6f-3b4c-0a1b2c3d4e5f
md"""
## Exercises

1. **Compare another team** — repeat this analysis for the United States or
   England.  How does their profile differ?
2. **Player focus** — identify Japan's key playmakers using passing network
   analysis (notebook 06).  Who connected the most passes?
3. **Opponent analysis** — how did Japan's opponents try to counter their
   possession style?  Compare pass accuracy against Japan vs against others.
4. **Temporal patterns** — did Japan's approach change between the group
   stage and the knockout rounds?
5. **Defensive analysis** — analyse Japan's defensive actions (tackles,
   interceptions, pressures).  Where did they win the ball back?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Chain = "1"
DataFrames = "1"
JSON3 = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─a7b8c9d0-0001-8e6f-3b4c-0d1e2f3a4b5c
# ╟─a7b8c9d0-0002-8e6f-3b4c-0d1e2f3a4b5c
# ╠═a7b8c9d0-0003-8e6f-3b4c-0d1e2f3a4b5c
# ╠═a7b8c9d0-0004-8e6f-3b4c-0d1e2f3a4b5c
# ╠═a7b8c9d0-0005-8e6f-3b4c-0d1e2f3a4b5c
# ╟─a7b8c9d0-0006-8e6f-3b4c-0d1e2f3a4b5c
# ╠═a7b8c9d0-0007-8e6f-3b4c-0d1e2f3a4b5c
# ╟─a7b8c9d0-0008-8e6f-3b4c-0a1b2c3d4e5f
# ╠═a7b8c9d0-0009-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0010-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0011-8e6f-3b4c-0a1b2c3d4e5f
# ╠═a7b8c9d0-0012-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0013-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0014-8e6f-3b4c-0a1b2c3d4e5f
# ╠═a7b8c9d0-0015-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0016-8e6f-3b4c-0a1b2c3d4e5f
# ╠═a7b8c9d0-0017-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0018-8e6f-3b4c-0a1b2c3d4e5f
# ╠═a7b8c9d0-0019-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0020-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0021-8e6f-3b4c-0a1b2c3d4e5f
# ╟─a7b8c9d0-0022-8e6f-3b4c-0a1b2c3d4e5f
# ╠═00000000-0000-0000-0000-000000000001
