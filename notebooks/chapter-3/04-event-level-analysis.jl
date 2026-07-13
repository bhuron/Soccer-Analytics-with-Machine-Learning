### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ d4e5f6a7-0003-5b3c-0e2d-9f0e7d6c5b4a
begin
	using JSON3
	using DataFrames
	using Statistics
	using Chain: @chain
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ d4e5f6a7-0001-5b3c-0e2d-9f0e7d6c5b4a
md"""
# Event-Level Analysis & Visualization

**Chapter 3 · Exploratory Data Analysis in Soccer**

## What you'll learn

- Load and explore event-level data from StatsBomb
- Filter events by type, team, and player
- Group and aggregate events into meaningful summaries
- Calculate pass accuracy, shot conversion rate, and expected goals
- Create visualizations for event patterns
"""

# ╔═╡ d4e5f6a7-0002-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Imports & setup
"""

# ╔═╡ d4e5f6a7-0004-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Loading a single match

Event data lives in one JSON file per match.  Each file holds every on-ball
action — passes, shots, tackles, carries — that occurred during that match.

Let's start with one match to understand the structure, then scale up.
"""

# ╔═╡ d4e5f6a7-0005-5b3c-0e2d-9f0e7d6c5b4a
function flatten_dict(d::AbstractDict; prefix="")
    flat = Dict{String,Any}()
    for (k, v) in d
        new_key = isempty(prefix) ? string(k) : "$prefix.$k"
        if v isa AbstractDict
            merge!(flat, flatten_dict(v; prefix=new_key))
        else
            flat[new_key] = v
        end
    end
    return flat
end

# ╔═╡ d4e5f6a7-0006-5b3c-0e2d-9f0e7d6c5b4a
begin
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")
	event_file = joinpath(DATA_DIR, "events", "22921.json")
	isfile(event_file) || error("File not found: $event_file")

	raw = JSON3.read(read(event_file, String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)
	rows = [let row = Dict{String,Any}()
		for k in all_keys; row[k] = get(d, k, missing); end; row
	end for d in dicts]

	global events = DataFrame(rows)
	events.match_id .= 22921
	md"""Loaded **$(nrow(events)) events** from match 22921 (France vs Korea Republic) across $(ncol(events)) columns."""
end

# ╔═╡ d4e5f6a7-0007-5b3c-0e2d-9f0e7d6c5b4a
first(events, 5)

# ╔═╡ d4e5f6a7-0008-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Understanding event types

Each event has a `type.name` that describes the action.
"""

# ╔═╡ d4e5f6a7-0009-5b3c-0e2d-9f0e7d6c5b4a
let
	counts = @chain events begin
		groupby(_, "type.name"; sort=true)
		combine(nrow => :count)
		sort(_, :count, rev=true)
		first(_, 10)
	end

	n = nrow(counts)
	labs = reverse(Vector(counts[!, "type.name"]))
	vals = reverse(Vector(counts.count))
	p = plot(xlims=(0, maximum(vals) + 100), ylims=(0.5, n + 0.5),
		legend=false, title="Top 10 Event Types in Match",
		xlabel="Count", yticks=(1:n, labs))
	for i in 1:n
		plot!(p, [0, vals[i]], [i, i], linewidth=12, color=:steelblue, legend=false)
	end
	p
end

# ╔═╡ d4e5f6a7-0010-5b3c-0e2d-9f0e7d6c5b4a
md"""
**What this tells us:** passes dominate (~60–70% of events), followed by ball
receipts, carries, and pressure events.  Shots and goals are rare but decisive.
"""

# ╔═╡ d4e5f6a7-0011-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Filtering — focusing on specific events

Let's look at France's passes: filter by type and team.
"""

# ╔═╡ d4e5f6a7-0012-5b3c-0e2d-9f0e7d6c5b4a
let
	france_passes = subset(events,
		"type.name" => ByRow(==("Pass")),
		"team.name" => ByRow(==("France Women's")))
	first(france_passes[!, ["minute", "second", "player.name", "type.name", "pass.outcome.name"]], 10)
end

# ╔═╡ d4e5f6a7-0013-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Understanding pass outcomes

A **completed** pass has *missing* `pass.outcome.name`.  Any non-missing value
("Incomplete", "Out", "Pass Offside") means the pass failed.
"""

# ╔═╡ d4e5f6a7-0014-5b3c-0e2d-9f0e7d6c5b4a
let
	all_passes = subset(events, "type.name" => ByRow(==("Pass")))
	total = nrow(all_passes)
	completed = count(ismissing, all_passes[!, "pass.outcome.name"])
	accuracy = completed / total

	md"""
	| Metric | Value |
	|---|---|
	| Total passes | **$total** |
	| Completed | **$completed** |
	| Incomplete | **$(total - completed)** |
	| Pass accuracy | **$(round(100 * accuracy; digits=1))%** |
	"""
end

# ╔═╡ d4e5f6a7-0015-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Multi-match event loading

Load events from 5 matches, stack them into one big DataFrame.
"""

# ╔═╡ d4e5f6a7-0016-5b3c-0e2d-9f0e7d6c5b4a
begin
	match_file = joinpath(DATA_DIR, "matches", "72", "30.json")
	match_raw = JSON3.read(read(match_file, String))
	match_ids = [d["match_id"] for d in match_raw][1:5]

	all_dfs = []
	for mid in match_ids
		evt_file = joinpath(DATA_DIR, "events", "$mid.json")
		raw2 = JSON3.read(read(evt_file, String))
		dicts2 = flatten_dict.(raw2)
		keys_set = union((keys(d) for d in dicts2)...)
		r2 = [let row = Dict{String,Any}()
			for k in keys_set; row[k] = get(d, k, missing); end; row
		end for d in dicts2]
		df = DataFrame(r2)
		df.match_id .= mid
		push!(all_dfs, df)
	end

	global multi_events = vcat(all_dfs...; cols=:union)
	md"""Loaded **$(nrow(multi_events)) events** from **$(length(match_ids)) matches** across $(ncol(multi_events)) columns."""
end

# ╔═╡ d4e5f6a7-0017-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Per-match pass summaries

For each (match, team) pair: pass attempts, completions, accuracy.
"""

# ╔═╡ d4e5f6a7-0018-5b3c-0e2d-9f0e7d6c5b4a
begin
	all_passes = subset(multi_events, "type.name" => ByRow(==("Pass")))
	all_passes.completed = ismissing.(all_passes[!, "pass.outcome.name"])

	global per_match = @chain all_passes begin
		groupby(_, ["match_id", "team.name"])
		combine(
			"type.name" => length => :attempted,
			:completed   => sum   => :completed)
		transform(_, [:attempted, :completed] =>
			ByRow((a, c) -> c / a) => :pass_accuracy)
	end
	first(per_match, 10)
end

# ╔═╡ d4e5f6a7-0019-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Pass accuracy by team
"""

# ╔═╡ d4e5f6a7-0020-5b3c-0e2d-9f0e7d6c5b4a
let
	team_acc = @chain per_match begin
		groupby(_, "team.name")
		combine(:pass_accuracy => mean => :avg_accuracy)
		sort(_, :avg_accuracy, rev=true)
	end

	n = nrow(team_acc)
	labs = reverse(Vector(team_acc[!, "team.name"]))
	vals = reverse(Vector(team_acc.avg_accuracy))
	p = plot(xlims=(0.5, 1.0), ylims=(0.5, n + 0.5),
		legend=false, title="Average Pass Accuracy by Team",
		xlabel="Pass Accuracy", yticks=(1:n, labs))
	for i in 1:n
		plot!(p, [0, vals[i]], [i, i], linewidth=10, color=:steelblue, legend=false)
	end
	p
end

# ╔═╡ d4e5f6a7-0021-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Analyzing shots and expected goals

Passing controls the ball.  Shots decide the match.
"""

# ╔═╡ d4e5f6a7-0022-5b3c-0e2d-9f0e7d6c5b4a
let
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")

	total_shots = nrow(shots)
	n_goals = sum(shots.is_goal)
	conv_rate = n_goals / total_shots

	xg_col = "shot.statsbomb_xg"
	has_xg = xg_col in names(shots)
	xg_msg = if has_xg
		avg_xg = mean(skipmissing(shots[!, xg_col]))
		total_xg = sum(skipmissing(shots[!, xg_col]))
		"| Average xG / shot | **$(round(avg_xg; digits=3))** |\n| Total xG | **$(round(total_xg; digits=2))** |"
	else
		"| xG data | *not available* |"
	end

	md"""
	| Metric | Value |
	|---|---|
	| Total shots | **$total_shots** |
	| Goals | **$n_goals** |
	| Conversion rate | **$(round(100 * conv_rate; digits=1))%** |
	$xg_msg
	"""
end

# ╔═╡ d4e5f6a7-0023-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Per-match shot statistics

Shots, goals, and (if available) xG for each team in each match.
"""

# ╔═╡ d4e5f6a7-0024-5b3c-0e2d-9f0e7d6c5b4a
begin
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")
	xg_col = "shot.statsbomb_xg"

	global per_match_shots
	if xg_col in names(shots)
		per_match_shots = @chain shots begin
			groupby(_, ["match_id", "team.name"])
			combine(
				"type.name" => length => :shots,
				:is_goal    => sum   => :goals,
				xg_col      => (x -> sum(skipmissing(x))) => :xg)
		end
	else
		per_match_shots = @chain shots begin
			groupby(_, ["match_id", "team.name"])
			combine(
				"type.name" => length => :shots,
				:is_goal    => sum   => :goals)
		end
	end
	first(per_match_shots, 10)
end

# ╔═╡ d4e5f6a7-0025-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Team-level shot summary
"""

# ╔═╡ d4e5f6a7-0026-5b3c-0e2d-9f0e7d6c5b4a
let
	team_summary = @chain per_match_shots begin
		groupby(_, "team.name")
		combine(:shots => sum => :total_shots,
		        :goals => sum => :total_goals,
		        (:shots, :goals) => ByRow((s, g) -> g / s) => :conversion_rate)
		sort(_, :total_goals, rev=true)
	end
	team_summary
end

# ╔═╡ d4e5f6a7-0027-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Shot volume vs. goals scored

More shots → more goals?  Let's check.
"""

# ╔═╡ d4e5f6a7-0028-5b3c-0e2d-9f0e7d6c5b4a
let
	avg = @chain per_match_shots begin
		groupby(_, "team.name")
		combine(:shots => mean => :shots_per_match,
		        :goals => mean => :goals_per_match)
	end

	x = avg.shots_per_match
	y = avg.goals_per_match

	X = hcat(ones(length(x)), x)
	coeffs = X \ y
	x_fit = range(minimum(x), maximum(x); length=50)
	y_fit = coeffs[1] .+ coeffs[2] .* x_fit

	scatter(x, y, legend=false, markersize=8, color=:darkblue,
		title="Shot Volume vs Goals Scored",
		xlabel="Shots per Match", ylabel="Goals per Match")
	plot!(x_fit, y_fit, color=:blue, linestyle=:dash, linewidth=2, label="Trend")
end

# ╔═╡ d4e5f6a7-0029-5b3c-0e2d-9f0e7d6c5b4a
md"""
**What this tells us:** a clear positive correlation.  Teams above the trend
line over-perform (clinical finishing); teams below need better conversion.
"""

# ╔═╡ d4e5f6a7-0030-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Distribution of shot quality (xG)

Most shots are low-probability — high-quality chances are rare.
"""

# ╔═╡ d4e5f6a7-0031-5b3c-0e2d-9f0e7d6c5b4a
let
	shots2 = subset(multi_events, "type.name" => ByRow(==("Shot")))
	if "shot.statsbomb_xg" in names(shots2)
		xg_vals = collect(skipmissing(shots2[!, "shot.statsbomb_xg"]))
		histogram(xg_vals, bins=25, color=:salmon, alpha=0.8, legend=false,
			title="Distribution of Shot Quality (xG)",
			xlabel="Shot xG (Expected Goals)", ylabel="Count")
	else
		md"*(xG data not available)*"
	end
end

# ╔═╡ d4e5f6a7-0032-5b3c-0e2d-9f0e7d6c5b4a
md"""
**What this tells us:** most shots carry xG < 0.1.  Chances above 0.3 xG are
rare and precious.
"""

# ╔═╡ d4e5f6a7-0033-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Recap

1. Loaded event-level data from StatsBomb JSON files
2. Filtered events by type, team, and player
3. Grouped and aggregated to create summaries
4. Calculated pass accuracy, shot conversion rate, xG totals
5. Created bar charts, scatter plots, and histograms

## Up next

Advanced visualizations (shot maps, heatmaps), passing networks, and a case
study of Japan at WWC 2019.
"""

# ╔═╡ d4e5f6a7-0034-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Exercises

1. **Pass length** — calculate the average pass length for each team using
   the `pass.length` column.
2. **Shot distance** — analyse the distribution of shot distances from goal.
3. **Player leaderboard** — find the top 10 players by passes completed.
4. **Time-based analysis** — shot frequency by 15-minute intervals.
5. **Defensive actions** — count tackles, interceptions, clearances by team.
"""

# ╔═╡ d4e5f6a7-0035-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Solutions
"""

# ╔═╡ d4e5f6a7-0036-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 1 — Average pass length per team

The `pass.length` column gives distance in metres.
"""

# ╔═╡ d4e5f6a7-0037-5b3c-0e2d-9f0e7d6c5b4a
@chain multi_events begin
	subset(_, "type.name" => ByRow(==("Pass")))
	dropmissing(_, "pass.length")
	groupby(_, "team.name")
	combine("pass.length" => mean => :avg_pass_length)
	sort(_, :avg_pass_length, rev=true)
end

# ╔═╡ d4e5f6a7-0038-5b3c-0e2d-9f0e7d6c5b4a
md"""
Longer passes → direct play.  Shorter passes → possession-oriented style.
"""

# ╔═╡ d4e5f6a7-0039-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 2 — Shot distance

Distance from goal centre ``(120,40)``: ``\sqrt{(120-x)^2 + (40-y)^2}``.
"""

# ╔═╡ d4e5f6a7-0040-5b3c-0e2d-9f0e7d6c5b4a
let
	shots2 = subset(multi_events, "type.name" => ByRow(==("Shot")))
	get_x(loc) = (ismissing(loc) || loc isa Missing) ? missing : loc[1]
	get_y(loc) = (ismissing(loc) || loc isa Missing) ? missing : loc[2]
	shots2.x = get_x.(shots2.location)
	shots2.y = get_y.(shots2.location)
	shots2.distance = sqrt.((120 .- shots2.x).^2 .+ (40 .- shots2.y).^2)
	clean = dropmissing(shots2, :distance)

	histogram(clean.distance, bins=20, color=:darkred, alpha=0.8,
		legend=false, title="Distribution of Shot Distances",
		xlabel="Distance from Goal (m)", ylabel="Count")
end

# ╔═╡ d4e5f6a7-0041-5b3c-0e2d-9f0e7d6c5b4a
md"""
Most shots from inside or just outside the box (10–25 m).
"""

# ╔═╡ d4e5f6a7-0042-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 3 — Top 10 players by passes completed
"""

# ╔═╡ d4e5f6a7-0043-5b3c-0e2d-9f0e7d6c5b4a
@chain multi_events begin
	subset(_, "type.name" => ByRow(==("Pass")))
	subset(_, "pass.outcome.name" => ByRow(ismissing))
	groupby(_, "player.name")
	combine(nrow => :completed_passes)
	sort(_, :completed_passes, rev=true)
	first(_, 10)
end

# ╔═╡ d4e5f6a7-0044-5b3c-0e2d-9f0e7d6c5b4a
md"""
These are the midfield organisers — the players who dictate tempo.
"""

# ╔═╡ d4e5f6a7-0045-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 4 — Shot frequency by 15-minute interval
"""

# ╔═╡ d4e5f6a7-0046-5b3c-0e2d-9f0e7d6c5b4a
let
	shots3 = subset(multi_events, "type.name" => ByRow(==("Shot")))
	shots3.interval = fld.(shots3.minute, 15) .+ 1
	shots3.interval_label = string.((shots3.interval .- 1) .* 15, "-", shots3.interval .* 15, "'")

	int_df = @chain shots3 begin
		groupby(_, "interval_label"; sort=true)
		combine(nrow => :shots)
	end

	n = nrow(int_df)
	xs = Vector(int_df[!, "interval_label"])
	ys = Vector(int_df.shots)
	p = plot(xlims=(0.5, n + 0.5), ylims=(0, maximum(ys) + 5),
		legend=false, title="Shots per 15-Minute Interval",
		xlabel="Match Period", ylabel="Number of Shots",
		xticks=(1:n, xs), xrotation=45)
	for i in 1:n
		plot!(p, [i, i], [0, ys[i]], linewidth=20, color=:purple, legend=false)
	end
	p
end

# ╔═╡ d4e5f6a7-0047-5b3c-0e2d-9f0e7d6c5b4a
md"""
Shot frequency often peaks just before half-time and at the end.
"""

# ╔═╡ d4e5f6a7-0048-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 5 — Defensive actions by team

Count tackles (Duel), interceptions, and clearances.
"""

# ╔═╡ d4e5f6a7-0049-5b3c-0e2d-9f0e7d6c5b4a
let
	defensive_types = ["Duel", "Interception", "Clearance"]
	def_actions = subset(multi_events, "type.name" => ByRow(t -> t in defensive_types))

	@chain def_actions begin
		groupby(_, ["team.name", "type.name"])
		combine(nrow => :count)
		unstack(_, "type.name", "team.name", :count; fill=0)
	end
end

# ╔═╡ d4e5f6a7-0050-5b3c-0e2d-9f0e7d6c5b4a
md"""
High duel counts → aggressive pressing.  High interceptions → good reading
of the game.
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
# ╟─d4e5f6a7-0001-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0002-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0003-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0004-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0005-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0006-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0007-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0008-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0009-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0010-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0011-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0012-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0013-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0014-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0015-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0016-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0017-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0018-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0019-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0020-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0021-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0022-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0023-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0024-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0025-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0026-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0027-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0028-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0029-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0030-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0031-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0032-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0033-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0034-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0035-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0036-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0037-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0038-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0039-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0040-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0041-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0042-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0043-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0044-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0045-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0046-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0047-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0048-5b3c-0e2d-9f0e7d6c5b4a
# ╠═d4e5f6a7-0049-5b3c-0e2d-9f0e7d6c5b4a
# ╟─d4e5f6a7-0050-5b3c-0e2d-9f0e7d6c5b4a
# ╠═00000000-0000-0000-0000-000000000001
