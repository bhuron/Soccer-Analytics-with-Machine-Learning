### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

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

# ╔═╡ d4e5f6a7-0004-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Loading a single match

Event data lives in one JSON file per match.  Each file holds every on-ball
action — passes, shots, tackles, carries — that occurred during that match.

Let's start with one match to understand the structure, then scale up.
"""

# ╔═╡ d4e5f6a7-0005-5b3c-0e2d-9f0e7d6c5b4a
"""
    flatten_dict(d::AbstractDict; prefix="")

Recursively flatten a nested dictionary, joining keys with `"."`.
Arrays and other non-dict values are left as-is.
"""
function flatten_dict(d::AbstractDict; prefix="")
    flat = Dict{String,Any}()
    for (k, v) in d
        new_key = isempty(prefix) ? string(k) : "$(prefix).$(k)"
        if v isa AbstractDict
            merge!(flat, flatten_dict(v; prefix=new_key))
        else
            flat[new_key] = v
        end
    end
    return flat
end

# ╔═╡ d4e5f6a7-0006-5b3c-0e2d-9f0e7d6c5b4a
let
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")
	event_file = joinpath(DATA_DIR, "events", "22921.json")
	isfile(event_file) || error("File not found: $event_file. Clone https://github.com/statsbomb/open-data")

	raw = JSON3.read(read(event_file, String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)

	rows = [let row = Dict{String,Any}()
		for k in all_keys
			row[k] = get(d, k, missing)
		end
		row
	end for d in dicts]

	global events = DataFrame(rows)
	events.match_id .= 22921
	md"""Loaded **$(nrow(events)) events** from match 22921 (France vs Korea Republic) across $(ncol(events)) columns."""
end

# ╔═╡ d4e5f6a7-0007-5b3c-0e2d-9f0e7d6c5b4a
# Peek at the first rows and key columns
first(events, 5)

# ╔═╡ d4e5f6a7-0008-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Understanding event types

Each event has a `type.name` that describes the action.  Let's see which
types dominate the event log.
"""

# ╔═╡ d4e5f6a7-0009-5b3c-0e2d-9f0e7d6c5b4a
let
	counts = @chain events begin
		groupby(_, "type.name"; sort=true)
		combine(nrow => :count)
		sort(_, :count, rev=true)
		first(_, 10)
	end

	# Horizontal bar chart — top 10 event types
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
**What this tells us:** passes dominate (~60–70 % of events), followed by ball
receipts, carries, and pressure events.  Shots and goals are rare but
decisive — exactly what you'd expect in a typical soccer match.
"""

# ╔═╡ d4e5f6a7-0011-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Filtering — focusing on specific events

The simplest analysis is filtering: select only the events we care about.
Let's look at France's passes.
"""

# ╔═╡ d4e5f6a7-0012-5b3c-0e2d-9f0e7d6c5b4a
let
	col_type = "type.name"
	col_team = "team.name"
	france_passes = @chain events begin
		subset(_, col_type => ByRow(==("Pass")), col_team => ByRow(==("France Women's")))
	end
	select(france_passes, :minute, :second, "player.name", "type.name", "pass.outcome.name") |>
		first .|> (x -> x(10))
end

# ╔═╡ d4e5f6a7-0013-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Understanding pass outcomes

In StatsBomb data, a **completed** pass has a *missing* value in
`pass.outcome.name`.  Any non-missing value ("Incomplete", "Out",
"Pass Offside", etc.) means the pass failed.
"""

# ╔═╡ d4e5f6a7-0014-5b3c-0e2d-9f0e7d6c5b4a
let
	all_passes = subset(events, "type.name" => ByRow(==("Pass")))
	total = nrow(all_passes)
	completed = count(ismissing, all_passes[!, "pass.outcome.name"])
	accuracy = completed / total

	outcome_counts = @chain all_passes begin
		groupby(_, "pass.outcome.name"; sort=true)
		combine(nrow => :count)
		sort(_, :count, rev=true)
	end

	md"""
	| Metric | Value |
	|---|---|
	| Total passes | **$total** |
	| Completed | **$completed** |
	| Incomplete | **$(total - completed)** |
	| Pass accuracy | **$(round(100 * accuracy, digits=1))%** |
	"""
end

# ╔═╡ d4e5f6a7-0015-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Multi-match event loading

Now let's load events from multiple matches and create per-match summaries.
We reuse the same flatten+align pattern, loop over match IDs, and stack.
"""

# ╔═╡ d4e5f6a7-0016-5b3c-0e2d-9f0e7d6c5b4a
let
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")
	# Load match list, take first 5 matches
	match_file = joinpath(DATA_DIR, "matches", "72", "30.json")
	match_raw = JSON3.read(read(match_file, String))
	match_ids = [d["match_id"] for d in match_raw][1:5]

	all_dfs = []
	for mid in match_ids
		evt_file = joinpath(DATA_DIR, "events", "$mid.json")
		raw = JSON3.read(read(evt_file, String))
		dicts = flatten_dict.(raw)
		keys_set = union((keys(d) for d in dicts)...)
		rows = [let row = Dict{String,Any}()
			for k in keys_set; row[k] = get(d, k, missing); end; row
		end for d in dicts]
		df = DataFrame(rows)
		df.match_id .= mid
		push!(all_dfs, df)
	end

	global multi_events = vcat(all_dfs...; cols=:union)
	md"""Loaded **$(nrow(multi_events)) events** from **$(length(match_ids)) matches** across $(ncol(multi_events)) columns."""
end

# ╔═╡ d4e5f6a7-0017-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Per-match pass summaries

For each (match, team) pair we compute pass attempts, completions, and
accuracy.
"""

# ╔═╡ d4e5f6a7-0018-5b3c-0e2d-9f0e7d6c5b4a
let
	all_passes = subset(multi_events, "type.name" => ByRow(==("Pass")))
	all_passes.completed = ismissing.(all_passes[!, "pass.outcome.name"])

	per_match = @chain all_passes begin
		groupby(_, ["match_id", "team.name"])
		combine(
			"type.name" => length => :attempted,
			:completed   => sum   => :completed,
		)
		transform(_, [:attempted, :completed] => ByRow((a, c) -> c / a) => :pass_accuracy)
	end

	first(per_match, 10)
end

# ╔═╡ d4e5f6a7-0019-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Pass accuracy by team

Let's average each team's per-match accuracy and compare.
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

Passing tells us who controls the ball.  Shots decide the match.  Let's
look at shot volume, conversion rates, and xG.
"""

# ╔═╡ d4e5f6a7-0022-5b3c-0e2d-9f0e7d6c5b4a
let
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")

	total_shots = nrow(shots)
	n_goals = sum(shots.is_goal)
	conv_rate = n_goals / total_shots

	# xG is available in StatsBomb open data
	xg_col = "shot.statsbomb_xg"
	has_xg = xg_col in names(shots)
	xg_msg = if has_xg
		avg_xg = mean(skipmissing(shots[!, xg_col]))
		total_xg = sum(skipmissing(shots[!, xg_col]))
		"| Average xG per shot | **$(round(avg_xg; digits=3))** |\n| Total xG | **$(round(total_xg; digits=2))** |"
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

A summary of shots, goals, and xG for each team in each match.
"""

# ╔═╡ d4e5f6a7-0024-5b3c-0e2d-9f0e7d6c5b4a
let
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")
	xg_col = "shot.statsbomb_xg"

	agg_spec = Dict(:shots => "type.name" => length, :goals => :is_goal => sum)
	if xg_col in names(shots)
		agg_spec[:xg] = xg_col => x -> sum(skipmissing(x))
	end

	per_match_shots = @chain shots begin
		groupby(_, ["match_id", "team.name"])
		combine(; agg_spec...)
	end

	first(per_match_shots, 10)
end

# ╔═╡ d4e5f6a7-0025-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Team-level shot summary

Roll up to tournament-level: total shots, goals, and xG for each team.
"""

# ╔═╡ d4e5f6a7-0026-5b3c-0e2d-9f0e7d6c5b4a
let
	team_summary = @chain per_match_shots begin
		groupby(_, "team.name")
		combine(:shots => sum, :goals => sum,
			(:shots, :goals) => ByRow((s, g) -> g / s) => :conversion_rate)
		sort(_, :goals_sum, rev=true)
	end
	team_summary
end

# ╔═╡ d4e5f6a7-0027-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Shot volume vs. goals scored

More shots → more goals?  Let's check with a scatter plot and trend line.
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

	# Linear fit: y = a + b*x
	X = hcat(ones(length(x)), x)
	coeffs = X \ y  # least-squares
	x_fit = range(minimum(x), maximum(x); length=50)
	y_fit = coeffs[1] .+ coeffs[2] .* x_fit

	scatter(x, y, legend=false, markersize=8, color=:darkblue,
		title="Shot Volume vs Goals Scored",
		xlabel="Shots per Match", ylabel="Goals per Match")
	plot!(x_fit, y_fit, color=:blue, linestyle=:dash, linewidth=2, label="Trend")
end

# ╔═╡ d4e5f6a7-0029-5b3c-0e2d-9f0e7d6c5b4a
md"""
**What this tells us:** a clear positive correlation — more shots leads to more
goals.  Teams above the trend line over-perform (clinical finishing); teams
below under-perform (poor conversion).  Both volume *and* efficiency matter.
"""

# ╔═╡ d4e5f6a7-0030-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Distribution of shot quality (xG)

Let's visualize how xG values are distributed.  Most shots are low-probability
— high-quality chances are rare.
"""

# ╔═╡ d4e5f6a7-0031-5b3c-0e2d-9f0e7d6c5b4a
let
	xg_col = "shot.statsbomb_xg"
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))

	if xg_col in names(shots)
		xg_vals = collect(skipmissing(shots[!, xg_col]))
		histogram(xg_vals, bins=25, color=:salmon, alpha=0.8, legend=false,
			title="Distribution of Shot Quality (xG)",
			xlabel="Shot xG (Expected Goals)", ylabel="Count")
	else
		md"*(xG data not available in this dataset)*"
	end
end

# ╔═╡ d4e5f6a7-0032-5b3c-0e2d-9f0e7d6c5b4a
md"""
**What this tells us:** most shots carry xG < 0.1 — low scoring probability.
Chances above 0.3 xG are rare and precious.  Teams that create more
high-xG opportunities will convert more goals.
"""

# ╔═╡ d4e5f6a7-0033-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Recap

In this notebook we:

1. Loaded event-level data from StatsBomb JSON files (single match, then
   multi-match)
2. Filtered events by type, team, and player
3. Grouped and aggregated to create meaningful summaries
4. Calculated key metrics: pass accuracy, shot conversion rate, xG totals
5. Created visualizations: bar charts, scatter plots, histograms

These are the fundamental building blocks of soccer analytics — from here
we can build predictive models and develop tactical insights.

## Up next

In the extras notebooks: advanced visualizations (shot maps, heatmaps),
passing network analysis, and a complete case study of Japan at WWC 2019.
"""

# ╔═╡ d4e5f6a7-0034-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Exercises

1. **Pass length** — calculate the average pass length for each team using
   the `pass.length` column.
2. **Shot distance** — analyse the distribution of shot distances (distance
   from goal can be approximated from `location` coordinates) and compare
   teams.
3. **Player leaderboard** — find the top 10 players by number of passes
   completed.
4. **Time-based analysis** — analyse how shot frequency changes throughout
   a match by 15-minute intervals.
5. **Defensive actions** — count tackles, interceptions, and clearances by
   team from the event data.
"""

# ╔═╡ d4e5f6a7-0035-5b3c-0e2d-9f0e7d6c5b4a
md"""
## Solutions

Try each exercise on your own first — the code below is just one way to
solve them.
"""

# ╔═╡ d4e5f6a7-0036-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 1 — Average pass length per team

The `pass.length` column gives the distance in metres.  We filter to passes
and average per team.
"""

# ╔═╡ d4e5f6a7-0037-5b3c-0e2d-9f0e7d6c5b4a
let
	result = @chain multi_events begin
		subset(_, "type.name" => ByRow(==("Pass")))
		dropmissing(_, "pass.length")
		groupby(_, "team.name")
		combine("pass.length" => mean => :avg_pass_length,
		        "pass.length" => (x -> round(mean(x); digits=1)) => :avg_rounded)
		sort(_, :avg_pass_length, rev=true)
	end
	select(result, :team_name, :avg_rounded)
end

# ╔═╡ d4e5f6a7-0038-5b3c-0e2d-9f0e7d6c5b4a
md"""
Longer passes are associated with more direct play; shorter passes suggest
a possession-oriented style.
"""

# ╔═╡ d4e5f6a7-0039-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 2 — Shot distance analysis

The `location` field is an array `[x, y]` where `(120, 40)` is the centre
of the opponent's goal.  We approximate distance as
``\sqrt{(120 - x)^2 + (40 - y)^2}``.
"""

# ╔═╡ d4e5f6a7-0040-5b3c-0e2d-9f0e7d6c5b4a
let
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))
	# Extract x coordinate from location array
	get_x(loc) = ismissing(loc) || loc isa Missing ? missing : loc[1]
	get_y(loc) = ismissing(loc) || loc isa Missing ? missing : loc[2]
	shots.x = get_x.(shots.location)
	shots.y = get_y.(shots.location)
	shots.distance = sqrt.((120 .- shots.x).^2 .+ (40 .- shots.y).^2)
	shots_clean = dropmissing(shots, :distance)

	histogram(shots_clean.distance, bins=20, color=:darkred, alpha=0.8,
		legend=false, title="Distribution of Shot Distances",
		xlabel="Distance from Goal (metres)", ylabel="Count")
end

# ╔═╡ d4e5f6a7-0041-5b3c-0e2d-9f0e7d6c5b4a
md"""
Most shots are taken from inside or just outside the box (10–25 m).  Shots
from beyond 30 m are rare — and rarely scored.
"""

# ╔═╡ d4e5f6a7-0042-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 3 — Top 10 players by passes completed

Count completed passes (where `pass.outcome.name` is missing) grouped by
player.
"""

# ╔═╡ d4e5f6a7-0043-5b3c-0e2d-9f0e7d6c5b4a
let
	passes = subset(multi_events, "type.name" => ByRow(==("Pass")))
	passes.completed = ismissing.(passes[!, "pass.outcome.name"])
	completed_only = subset(passes, :completed => ByRow(identity))

	result = @chain completed_only begin
		groupby(_, "player.name")
		combine(nrow => :completed_passes)
		sort(_, :completed_passes, rev=true)
		first(_, 10)
	end
	select(result, :player_name, :completed_passes)
end

# ╔═╡ d4e5f6a7-0044-5b3c-0e2d-9f0e7d6c5b4a
md"""
These are typically the team's midfield organisers — the players who dictate
tempo and connect defence to attack.
"""

# ╔═╡ d4e5f6a7-0045-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 4 — Shot frequency by 15-minute interval

Group shots by 15-minute bins to see when teams are most dangerous.
"""

# ╔═╡ d4e5f6a7-0046-5b3c-0e2d-9f0e7d6c5b4a
let
	shots = subset(multi_events, "type.name" => ByRow(==("Shot")))
	shots.interval = fld.(shots.minute, 15) .+ 1  # 1 = 0-14, 2 = 15-29, ...
	shots.interval_label = string.((shots.interval .- 1) .* 15, "-", shots.interval .* 15, "'")

	int_df = @chain shots begin
		groupby(_, :interval_label; sort=true)
		combine(nrow => :shots)
	end

	n = nrow(int_df)
	xs = Vector(int_df.interval_label)
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
Shot frequency often peaks just before half-time and at the end of the match
as teams push for goals.
"""

# ╔═╡ d4e5f6a7-0048-5b3c-0e2d-9f0e7d6c5b4a
md"""
### Exercise 5 — Defensive actions by team

Count tackles, interceptions, and clearances per team from the event data.
The event type names we're looking for are `"Duel"`, `"Interception"`, and
`"Clearance"`.
"""

# ╔═╡ d4e5f6a7-0049-5b3c-0e2d-9f0e7d6c5b4a
let
	defensive_types = ["Duel", "Interception", "Clearance"]
	def_actions = subset(multi_events, "type.name" => ByRow(t -> t in defensive_types))

	result = @chain def_actions begin
		groupby(_, ["team.name", "type.name"])
		combine(nrow => :count)
		unstack(_, "type.name", "team.name", :count; fill=0)
	end
	result
end

# ╔═╡ d4e5f6a7-0050-5b3c-0e2d-9f0e7d6c5b4a
md"""
Teams with high duel counts tend to play a more aggressive, pressing style.
High interception counts often indicate a team that reads the game well
defensively.
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
