### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ f6a7b8c9-0003-7d5e-2a4f-1b2c3d4e5f6a
begin
	using JSON3
	using DataFrames
	using Statistics
	using Chain: @chain
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=9)
end

# ╔═╡ f6a7b8c9-0001-7d5e-2a4f-1b2c3d4e5f6a
md"""
# Passing Networks

**Chapter 3 · Exploratory Data Analysis in Soccer**

## What you'll learn

- Calculate average player positions from event data
- Compute pass frequencies between player pairs
- Create passing network visualisations on a pitch
- Interpret tactical structures
- Identify key playmakers using network metrics

## What is a passing network?

- **Nodes** = players, positioned at their average pass location
- **Edges** = lines between players, thickness ∝ pass frequency
- **Node size** ∝ number of passes made

This reveals tactical structure and key relationships within a team.
"""

# ╔═╡ f6a7b8c9-0002-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Imports & setup
"""

# ╔═╡ f6a7b8c9-0004-7d5e-2a4f-1b2c3d4e5f6a
"""
    draw_pitch(; bgcolor, linecolor, linewidth)

Draw a soccer pitch in landscape orientation (120 × 80 StatsBomb coords).
Returns a Plots plot object.
"""
function draw_pitch(;
		bgcolor   = "#22312b",
		linecolor = "#efefef",
		linewidth = 1.5,
	)
	p = plot(xlims=(-2, 122), ylims=(-2, 82), aspect_ratio=:equal,
		legend=false, background_color=bgcolor, ticks=false, framestyle=:none)

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

# ╔═╡ f6a7b8c9-0005-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Load event data

France vs Korea Republic (match 22921).
"""

# ╔═╡ f6a7b8c9-0006-7d5e-2a4f-1b2c3d4e5f6a
begin
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")

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

	event_file = joinpath(DATA_DIR, "events", "22921.json")
	raw = JSON3.read(read(event_file, String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)
	rows = [let row = Dict{String,Any}()
		for k in all_keys; row[k] = get(d, k, missing); end; row
	end for d in dicts]

	global events = DataFrame(rows)
	events.match_id .= 22921
	md"""Loaded **$(nrow(events)) events** from match 22921."""
end

# ╔═╡ f6a7b8c9-0007-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Step 1 — Extract completed passes for one team

We keep only **completed** passes where `pass.outcome.name` is missing.
"""

# ╔═╡ f6a7b8c9-0008-7d5e-2a4f-1b2c3d4e5f6a
let
	team_name = "France Women's"

	global team_passes = @chain events begin
		subset(_,
			"type.name"         => ByRow(==("Pass")),
			"team.name"         => ByRow(==(team_name)),
			"pass.outcome.name" => ByRow(ismissing);
			skipmissing=true)
	end

	# Extract pass start coordinates
	team_passes.x = [(ismissing(loc) || loc === nothing) ? missing : loc[1] for loc in team_passes.location]
	team_passes.y = [(ismissing(loc) || loc === nothing) ? missing : loc[2] for loc in team_passes.location]

	# Keep only rows with passer, receiver, and valid coordinates
	global team_passes_clean = dropmissing(team_passes, ["player.name", "pass.recipient.name", "x", "y"])

	md"""**$team_name** completed **$(nrow(team_passes_clean)) passes** with full location + recipient data."""
end

# ╔═╡ f6a7b8c9-0009-7d5e-2a4f-1b2c3d4e5f6a
# Quick peek at the structure
first(select(team_passes_clean, "player.name", "pass.recipient.name", :x, :y), 5)

# ╔═╡ f6a7b8c9-0010-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Step 2 — Average player positions

For each player, their average (x, y) coordinate across all passes they made
is their "position" in the network.
"""

# ╔═╡ f6a7b8c9-0011-7d5e-2a4f-1b2c3d4e5f6a
let
	global player_pos = @chain team_passes_clean begin
		groupby(_, "player.name")
		combine(
			:x => mean => :avg_x,
			:y => mean => :avg_y,
			nrow => :num_passes)
		sort(_, :num_passes, rev=true)
	end
	first(player_pos, 10)
end

# ╔═╡ f6a7b8c9-0012-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Step 3 — Pass frequencies between players

Count how many times each pair of players connected.  Filter to
connections with at least **3 passes** to avoid noise.
"""

# ╔═╡ f6a7b8c9-0013-7d5e-2a4f-1b2c3d4e5f6a
let
	global pass_pairs = @chain team_passes_clean begin
		groupby(_, ["player.name", "pass.recipient.name"])
		combine(nrow => :pass_count)
		rename!(_, "player.name" => :passer, "pass.recipient.name" => :receiver)
		subset(_, :pass_count => ByRow(>=(3)); skipmissing=true)
	end

	md"""Found **$(nrow(pass_pairs)) significant passing connections** (≥ 3 passes)."""
end

# ╔═╡ f6a7b8c9-0014-7d5e-2a4f-1b2c3d4e5f6a
# Show top combinations
first(sort(pass_pairs, :pass_count, rev=true), 10)

# ╔═╡ f6a7b8c9-0015-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Step 4 — Draw the passing network

Nodes = players (position = average pass location, size ∝ pass count).
Edges = passing connections (thickness ∝ frequency).
"""

# ╔═╡ f6a7b8c9-0016-7d5e-2a4f-1b2c3d4e5f6a
let
	# Build position lookup: player name → (x, y)
	pos_lookup = Dict(row."player.name" => (row.avg_x, row.avg_y) for row in eachrow(player_pos))

	p = draw_pitch()

	# Draw edges (passing lines)
	for row in eachrow(pass_pairs)
		x1, y1 = get(pos_lookup, row.passer, (missing, missing))
		x2, y2 = get(pos_lookup, row.receiver, (missing, missing))
		if !ismissing(x1) && !ismissing(x2)
			lw = row.pass_count / 1.5  # scale thickness
			plot!(p, [x1, x2], [y1, y2], color=:white, alpha=0.25, lw=lw, legend=false)
		end
	end

	# Draw nodes (player positions)
	node_sizes = player_pos.num_passes .* 2.5
	scatter!(p, player_pos.avg_x, player_pos.avg_y,
		markersize=node_sizes, color=:lime, markerstrokecolor=:white,
		markerstrokewidth=1.5, legend=false, zorder=3)

	# Label top 11 players (starting XI)
	for row in eachrow(first(player_pos, 11))
		name_parts = split(row."player.name")
		label = length(name_parts) > 1 ? name_parts[end] : row."player.name"
		annotate!(p, row.avg_x, row.avg_y - 3,
			Plots.text(label, 8, :white, :center))
	end

	title!("Passing Network — France Women's")
	p
end

# ╔═╡ f6a7b8c9-0017-7d5e-2a4f-1b2c3d4e5f6a
md"""
**How to read this:**

1. **Node position** — where each player operated on average.  Reveals
   formation — defensive players lower, attackers higher, wide players
   near the touchlines.
2. **Node size** — passing volume.  Larger nodes = key distributors.
3. **Edge thickness** — how often two players connected.  Thick lines =
   preferred combinations.
4. **Isolated players** — few or no connections may indicate tactical
   issues or a player being marked out of the game.
"""

# ╔═╡ f6a7b8c9-0018-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Step 5 — Identify key playmakers

Quantify player importance by counting unique passing partners.
"""

# ╔═╡ f6a7b8c9-0019-7d5e-2a4f-1b2c3d4e5f6a
let
	# Count unique partners as passer
	outgoing = @chain pass_pairs begin
		groupby(_, :passer)
		combine(:receiver => length ∘ unique => :unique_targets)
		rename!(_, :passer => :player)
	end

	# Count unique partners as receiver
	incoming = @chain pass_pairs begin
		groupby(_, :receiver)
		combine(:passer => length ∘ unique => :unique_sources)
		rename!(_, :receiver => :player)
	end

	# Merge both into player_pos
	metrics = leftjoin(player_pos, outgoing, on="player.name" => :player)
	metrics = leftjoin(metrics, incoming, on="player.name" => :player)
	metrics.total_connections = coalesce.(metrics.unique_targets, 0) .+ coalesce.(metrics.unique_sources, 0)

	select(metrics, "player.name", :num_passes, :total_connections) |>
		df -> sort(df, :total_connections, rev=true) |>
		df -> first(df, 10)
end

# ╔═╡ f6a7b8c9-0020-7d5e-2a4f-1b2c3d4e5f6a
md"""
A high number of unique connections means the player links to many
teammates — they're the hub of the passing network.  These are typically
central midfielders or deep-lying playmakers.
"""

# ╔═╡ f6a7b8c9-0021-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Recap

1. Extracted completed passes for one team from event data
2. Computed average player positions from pass start locations
3. Built pass-frequency table between player pairs
4. Drew a complete passing network on a pitch — nodes, edges, labels
5. Quantified playmaker importance via unique connection counts

## Up next

- Compare passing networks between teams or halves
- Analyse how tactics shift in different game states
- Explore defensive action networks
"""

# ╔═╡ f6a7b8c9-0022-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Exercises

1. **Compare two teams** — create side-by-side passing networks for both
   teams in the same match.
2. **First half vs second half** — separate networks for periods 1 and 2.
3. **Progressive passes** — filter for forward passes only (x-end > x-start)
   and build a network.
4. **Weighted edges by xT** — if you compute expected-threat values, use
   them to colour edges instead of raw pass count.
5. **Defensive network** — create a network of interceptions and clearances
   between players.
"""

# ╔═╡ f6a7b8c9-0023-7d5e-2a4f-1b2c3d4e5f6a
md"""
## Solutions
"""

# ╔═╡ f6a7b8c9-0024-7d5e-2a4f-1b2c3d4e5f6a
md"""
### Exercise 1 — Side-by-side team networks

Build a network for each team and display them next to each other.
"""

# ╔═╡ f6a7b8c9-0025-7d5e-2a4f-1b2c3d4e5f6a
function build_network(events, team_name; min_passes=3)
	# Filter completed passes for this team
	passes = @chain events begin
		subset(_,
			"type.name"         => ByRow(==("Pass")),
			"team.name"         => ByRow(==(team_name)),
			"pass.outcome.name" => ByRow(ismissing);
			skipmissing=true)
	end
	passes.x = [(ismissing(l) || l === nothing) ? missing : l[1] for l in passes.location]
	passes.y = [(ismissing(l) || l === nothing) ? missing : l[2] for l in passes.location]
	clean = dropmissing(passes, ["player.name", "pass.recipient.name", "x", "y"])

	# Player positions
	positions = @chain clean begin
		groupby(_, "player.name")
		combine(:x => mean => :avg_x, :y => mean => :avg_y, nrow => :num_passes)
	end

	# Pass pairs
	pairs = @chain clean begin
		groupby(_, ["player.name", "pass.recipient.name"])
		combine(nrow => :pass_count)
		rename!(_, "player.name" => :passer, "pass.recipient.name" => :receiver)
		subset(_, :pass_count => ByRow(>=(min_passes)); skipmissing=true)
	end

	# Draw
	pos_dict = Dict(r."player.name" => (r.avg_x, r.avg_y) for r in eachrow(positions))
	plt = draw_pitch()
	for r in eachrow(pairs)
		x1, y1 = get(pos_dict, r.passer, (missing, missing))
		x2, y2 = get(pos_dict, r.receiver, (missing, missing))
		if !ismissing(x1) && !ismissing(x2)
			plot!(plt, [x1, x2], [y1, y2], color=:white, alpha=0.25,
				lw=r.pass_count / 1.5, legend=false)
		end
	end
	scatter!(plt, positions.avg_x, positions.avg_y,
		markersize=positions.num_passes .* 2.5, color=:lime,
		markerstrokecolor=:white, markerstrokewidth=1.5, legend=false)
	title!(plt, team_name)
	plt
end

# ╔═╡ f6a7b8c9-0026-7d5e-2a4f-1b2c3d4e5f6a
let
	teams = unique(skipmissing(events[!, "team.name"]))
	plots = [build_network(events, t) for t in teams if !ismissing(t)]
	plot(plots..., layout=(1, length(plots)), size=(1400, 500))
end

# ╔═╡ f6a7b8c9-0027-7d5e-2a4f-1b2c3d4e5f6a
md"""
### Exercise 2 — First half vs second half

Build separate networks for minutes 0–45 and 45–90.
"""

# ╔═╡ f6a7b8c9-0028-7d5e-2a4f-1b2c3d4e5f6a
let
	team_name = "France Women's"
	# Tag each event by half
	events.half = ifelse.(events.period .== 1, "First Half", "Second Half")

	plots_half = []
	for half in ["First Half", "Second Half"]
		half_events = subset(events, :half => ByRow(==(half)); skipmissing=true)
		plt = build_network(half_events, team_name; min_passes=2)
		title!(plt, "$team_name — $half")
		push!(plots_half, plt)
	end
	plot(plots_half..., layout=(1, 2), size=(1400, 500))
end

# ╔═╡ f6a7b8c9-0029-7d5e-2a4f-1b2c3d4e5f6a
md"""
### Exercise 3 — Progressive passes only

A progressive pass moves the ball toward the opponent's goal.  In StatsBomb
coordinates, that means `end_location[1] > start_location[1]` (higher x =
closer to opponent's goal).
"""

# ╔═╡ f6a7b8c9-0030-7d5e-2a4f-1b2c3d4e5f6a
let
	team_name = "France Women's"
	passes = subset(events,
		"type.name"         => ByRow(==("Pass")),
		"team.name"         => ByRow(==(team_name)),
		"pass.outcome.name" => ByRow(ismissing);
		skipmissing=true)

	# Extract start and end x coordinates
	passes.start_x = [(ismissing(l) || l === nothing) ? missing : l[1] for l in passes.location]
	passes.start_y = [(ismissing(l) || l === nothing) ? missing : l[2] for l in passes.location]
	passes.end_x = [(ismissing(l) || l === nothing) ? missing : l[1] for l in passes[!, "pass.end_location"]]
	clean = dropmissing(passes, ["player.name", "pass.recipient.name", "start_x", "start_y", "end_x"])

	# Only forward passes
	prog = subset(clean, :end_x => ByRow(x -> x > clean.start_x); skipmissing=true)

	# Temporarily rename columns so build_network works
	rename!(prog, :start_x => :x, :start_y => :y)
	prog_events = prog  # pass to build_network
	# Actually the build_network function re-extracts from events...
	# Let's just reuse the function with modified data
	# We need to: temporarily swap .x/.y on the DataFrame
	# For simplicity, rebuild the pipeline inline

	pos = @chain prog begin
		groupby(_, "player.name")
		combine(:x => mean => :avg_x, :y => mean => :avg_y, nrow => :num_passes)
	end

	pairs = @chain prog begin
		groupby(_, ["player.name", "pass.recipient.name"])
		combine(nrow => :pass_count)
		rename!(_, "player.name" => :passer, "pass.recipient.name" => :receiver)
		subset(_, :pass_count => ByRow(>=(2)); skipmissing=true)
	end

	pos_dict = Dict(r."player.name" => (r.avg_x, r.avg_y) for r in eachrow(pos))
	plt = draw_pitch()
	for r in eachrow(pairs)
		x1, y1 = get(pos_dict, r.passer, (missing, missing))
		x2, y2 = get(pos_dict, r.receiver, (missing, missing))
		if !ismissing(x1)
			lw = r.pass_count / 1.5
			plot!(plt, [x1, x2], [y1, y2], color=:gold, alpha=0.3, lw=lw, legend=false)
		end
	end
	scatter!(plt, pos.avg_x, pos.avg_y,
		markersize=pos.num_passes .* 3, color=:lime,
		markerstrokecolor=:white, markerstrokewidth=1.5)
	title!(plt, "$team_name — Progressive Passes Only")
	for r in eachrow(first(pos, 11))
		parts = split(r."player.name")
		label = length(parts) > 1 ? parts[end] : r."player.name"
		annotate!(plt, r.avg_x, r.avg_y - 3, Plots.text(label, 8, :white, :center))
	end
	plt
end

# ╔═╡ f6a7b8c9-0031-7d5e-2a4f-1b2c3d4e5f6a
md"""
The gold edges show only forward-moving passes.  This strips away
back-passes and sideways recycling, revealing the team's attacking
structure: who progresses the ball, and to whom.
"""

# ╔═╡ f6a7b8c9-0032-7d5e-2a4f-1b2c3d4e5f6a
md"""
### Exercise 5 — Defensive actions network

Instead of passes, connect players who made interceptions or clearances
to show the defensive structure.
"""

# ╔═╡ f6a7b8c9-0033-7d5e-2a4f-1b2c3d4e5f6a
let
	team_name = "France Women's"
	def_types = ["Interception", "Clearance"]
	def_actions = subset(events,
		"type.name" => ByRow(t -> t in def_types),
		"team.name" => ByRow(==(team_name));
		skipmissing=true)
	def_actions.x = [(ismissing(l) || l === nothing) ? missing : l[1] for l in def_actions.location]
	def_actions.y = [(ismissing(l) || l === nothing) ? missing : l[2] for l in def_actions.location]
	clean = dropmissing(def_actions, ["player.name", "x", "y"])

	pos = @chain clean begin
		groupby(_, "player.name")
		combine(:x => mean => :avg_x, :y => mean => :avg_y, nrow => :actions)
	end

	plt = draw_pitch()
	scatter!(plt, pos.avg_x, pos.avg_y,
		markersize=pos.actions .* 8, color=:orangered, alpha=0.8,
		markerstrokecolor=:white, markerstrokewidth=1.5, legend=false)
	for r in eachrow(pos)
		parts = split(r."player.name")
		label = length(parts) > 1 ? parts[end] : r."player.name"
		annotate!(plt, r.avg_x, r.avg_y - 3, Plots.text(label, 7, :white, :center))
	end
	title!(plt, "$team_name — Defensive Actions")
	plt
end

# ╔═╡ f6a7b8c9-0034-7d5e-2a4f-1b2c3d4e5f6a
md"""
Large orange nodes show players who made the most interceptions and
clearances — typically centre-backs and defensive midfielders.  This
is a simpler network (no edges), but the spatial distribution reveals
the team's defensive shape.
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
# ╟─f6a7b8c9-0001-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0002-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0003-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0004-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0005-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0006-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0007-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0008-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0009-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0010-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0011-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0012-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0013-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0014-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0015-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0016-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0017-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0018-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0019-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0020-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0021-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0022-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0023-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0024-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0025-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0026-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0027-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0028-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0029-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0030-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0031-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0032-7d5e-2a4f-1b2c3d4e5f6a
# ╠═f6a7b8c9-0033-7d5e-2a4f-1b2c3d4e5f6a
# ╟─f6a7b8c9-0034-7d5e-2a4f-1b2c3d4e5f6a
# ╠═00000000-0000-0000-0000-000000000001
