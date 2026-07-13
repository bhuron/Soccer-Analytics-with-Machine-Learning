### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ e5f6a7b8-0003-6c4d-1f3e-0a1b2c3d4e5f
begin
	using JSON3
	using DataFrames
	using Statistics
	using Chain: @chain
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=9)
end

# ╔═╡ e5f6a7b8-0001-6c4d-1f3e-0a1b2c3d4e5f
md"""
# Advanced Soccer Visualizations

**Chapter 3 · Exploratory Data Analysis in Soccer**

## What you'll learn

- Draw a professional soccer pitch with Plots.jl
- Create shot maps showing goals and misses
- Size shot markers by expected goals (xG)
- Build pass heatmaps with histogram2d and 2D histograms
- Customize pitch appearance and orientation
"""

# ╔═╡ e5f6a7b8-0002-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Imports & setup
"""

# ╔═╡ e5f6a7b8-0004-6c4d-1f3e-0a1b2c3d4e5f
"""
    draw_pitch(; color, linecolor, title)

Draw a soccer pitch in landscape orientation (120 × 80 StatsBomb coords).
Returns a Plots plot object ready for overlaid scatter/heatmap calls.

Use `orientation=:vertical` for a vertical (portrait) pitch.
"""
function draw_pitch(;
		bgcolor   = "#22312b",
		linecolor = "#efefef",
		linewidth = 1.5,
		title     = "",
	)
	p = plot(xlims=(-2, 122), ylims=(-2, 82), aspect_ratio=:equal,
		legend=false, title=title, background_color=bgcolor,
		ticks=false, framestyle=:none)

	# Outer boundary
	plot!(p, [0, 0, 120, 120, 0], [0, 80, 80, 0, 0],
		color=linecolor, linewidth=linewidth, legend=false)

	# Half-way line
	plot!(p, [60, 60], [0, 80], color=linecolor, linewidth=linewidth, legend=false)

	# Centre circle (radius ~9.15 m)
	θ = range(0, 2π; length=100)
	plot!(p, 60 .+ 9.15 .* cos.(θ), 40 .+ 9.15 .* sin.(θ),
		color=linecolor, linewidth=linewidth, legend=false)
	scatter!(p, [60], [40], color=linecolor, markersize=3, legend=false)

	# Left penalty area (x: 0→18, y: 18→62)
	plot!(p, [0, 18, 18, 0, 0], [18, 18, 62, 62, 18],
		color=linecolor, linewidth=linewidth, legend=false)

	# Right penalty area (x: 102→120, y: 18→62)
	plot!(p, [120, 102, 102, 120, 120], [18, 18, 62, 62, 18],
		color=linecolor, linewidth=linewidth, legend=false)

	# Left goal area (x: 0→6, y: 30→50)
	plot!(p, [0, 6, 6, 0, 0], [30, 30, 50, 50, 30],
		color=linecolor, linewidth=linewidth, legend=false)

	# Right goal area (x: 114→120, y: 30→50)
	plot!(p, [120, 114, 114, 120, 120], [30, 30, 50, 50, 30],
		color=linecolor, linewidth=linewidth, legend=false)

	# Penalty spots (x=12, x=108; y=40)
	scatter!(p, [12, 108], [40, 40], color=linecolor, markersize=4, legend=false)

	# Corner arcs (quarter-circles at each corner, radius ~1m)
	for (cx, cy, a0) in [(0, 0, 0), (120, 0, π/2), (120, 80, π), (0, 80, 3π/2)]
		θc = range(a0, a0 + π/2; length=20)
		plot!(p, cx .+ 1 .* cos.(θc), cy .+ 1 .* sin.(θc),
			color=linecolor, linewidth=linewidth, legend=false)
	end

	p
end

# ╔═╡ e5f6a7b8-0005-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Load event data

We'll work with one match: France vs Korea Republic (match 22921).
"""

# ╔═╡ e5f6a7b8-0006-6c4d-1f3e-0a1b2c3d4e5f
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
	md"""Loaded **$(nrow(events)) events** from match 22921 (France vs Korea Republic)."""
end

# ╔═╡ e5f6a7b8-0007-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Basic shot map

A shot map shows where shots were taken.  Goals are large bright markers;
misses are smaller, fainter dots.
"""

# ╔═╡ e5f6a7b8-0008-6c4d-1f3e-0a1b2c3d4e5f
let
	shots = subset(events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")

	# Extract x, y from the location array
	get_x(loc) = (ismissing(loc) || loc isa Missing || loc === nothing) ? missing : loc[1]
	get_y(loc) = (ismissing(loc) || loc isa Missing || loc === nothing) ? missing : loc[2]
	shots.x = get_x.(shots.location)
	shots.y = get_y.(shots.location)
	shots_clean = dropmissing(shots, [:x, :y])

	goals   = subset(shots_clean, :is_goal => ByRow(identity))
	misses  = subset(shots_clean, :is_goal => ByRow(!))

	p = draw_pitch(title="Shot Map — Goals vs Misses")

	scatter!(p, misses.x, misses.y,
		color=:red, markersize=5, alpha=0.5, label="Miss")
	scatter!(p, goals.x, goals.y,
		color=:lime, markersize=10, label="Goal",
		markerstrokecolor=:white, markerstrokewidth=1.5)

	p
end

# ╔═╡ e5f6a7b8-0009-6c4d-1f3e-0a1b2c3d4e5f
md"""
**What this tells us:**
- Green dots show goals (large, prominent)
- Red dots show misses (smaller, semi-transparent)
- Shot clusters near the penalty area indicate high-quality chances
- Shots from outside the box are more frequent but harder to convert
"""

# ╔═╡ e5f6a7b8-0010-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Shot map with xG sizing

Make marker area proportional to expected goals — larger dots mean
higher-quality chances.
"""

# ╔═╡ e5f6a7b8-0011-6c4d-1f3e-0a1b2c3d4e5f
let
	shots = subset(events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")
	shots.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(shots.location)
	shots.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(shots.location)
	shots.xg = shots[!, "shot.statsbomb_xg"]
	clean = dropmissing(shots, [:x, :y, :xg])
	clean.xg .= max.(clean.xg, 0.01)  # floor so zero-xG shots are still visible

	goals   = subset(clean, :is_goal => ByRow(identity))
	misses  = subset(clean, :is_goal => ByRow(!))

	p = draw_pitch(bgcolor="#1a1a2e", linecolor="#e0e0e0",
		title="Shot Map with xG Sizing")

	scale = 1200
	scatter!(p, misses.x, misses.y,
		markersize=misses.xg .* scale, color=:orangered,
		alpha=0.5, markerstrokecolor=:white, markerstrokewidth=0.5,
		label="Miss")
	scatter!(p, goals.x, goals.y,
		markersize=goals.xg .* scale, color=:lime,
		alpha=0.85, markerstrokecolor=:white, markerstrokewidth=1.5,
		label="Goal")

	p
end

# ╔═╡ e5f6a7b8-0012-6c4d-1f3e-0a1b2c3d4e5f
md"""
**What this tells us:**
- Larger circles = higher xG (better scoring chances)
- Large green circles = clinical finishing of good opportunities
- Large red circles = missed sitters — wasteful
- Small circles = low-probability efforts from distance
"""

# ╔═╡ e5f6a7b8-0013-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Pass location histogram2d heatmap

A histogram2d plot shows where passes originate.  Darker rectangular bins = higher pass
density.  Let's look at France's passing.
"""

# ╔═╡ e5f6a7b8-0014-6c4d-1f3e-0a1b2c3d4e5f
let
	passes = subset(events,
		"type.name" => ByRow(==("Pass")),
		"team.name"  => ByRow(==("France Women's")))
	passes.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(passes.location)
	passes.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(passes.location)
	clean = dropmissing(passes, [:x, :y])

	p = draw_pitch(bgcolor="#1a1a2e", linecolor="#aaaaaa",
		title="Pass Start Locations — France Women's")

	histogram2d!(p, clean.x, clean.y, gridsize=25, colorbar=true,
		colormap=:Blues, alpha=0.8, label="")

	p
end

# ╔═╡ e5f6a7b8-0015-6c4d-1f3e-0a1b2c3d4e5f
md"""
**What this tells us:**
- Darker rectangular bins = higher pass density
- Concentration in France's half shows buildup from the back
- Wide distribution across the pitch indicates use of full width
- The opponent's half shows fewer passes — the attacking third is
  contested space
"""

# ╔═╡ e5f6a7b8-0016-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Pass density heatmap — 2D histogram

A smoothed 2D histogram gives a different perspective on pass density,
using rectangular bins instead of rectangular bins.
"""

# ╔═╡ e5f6a7b8-0017-6c4d-1f3e-0a1b2c3d4e5f
let
	passes = subset(events,
		"type.name" => ByRow(==("Pass")),
		"team.name"  => ByRow(==("France Women's")))
	passes.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(passes.location)
	passes.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(passes.location)
	clean = dropmissing(passes, [:x, :y])

	p = draw_pitch(bgcolor="#1a1a2e", linecolor="#aaaaaa",
		title="Pass Density Heatmap — France Women's")

	histogram2d!(p, clean.x, clean.y, bins=(40, 27), colorbar=true,
		colormap=:YlOrRd, alpha=0.7, label="")

	p
end

# ╔═╡ e5f6a7b8-0018-6c4d-1f3e-0a1b2c3d4e5f
md"""
Rectangular bins give a different texture than rectangular bins — useful when you
want a smoother, more "continuous" look rather than discrete cells.
"""

# ╔═╡ e5f6a7b8-0019-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Vertical pitch orientation

A vertical (portrait) layout works well for presentations and social media.
We draw the pitch rotated 90° and replot the shot map.
"""

# ╔═╡ e5f6a7b8-0020-6c4d-1f3e-0a1b2c3d4e5f
let
	shots = subset(events, "type.name" => ByRow(==("Shot")))
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")
	shots.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(shots.location)
	shots.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(shots.location)
	clean = dropmissing(shots, [:x, :y])
	goals  = subset(clean, :is_goal => ByRow(identity))
	misses = subset(clean, :is_goal => ByRow(!))

	# Swap coordinates for vertical orientation: y → x, x → 80-y
	p = draw_pitch(title="Vertical Shot Map")

	# For vertical: we swap axes.  StatsBomb x (0→120) becomes the vertical
	# axis.  y (0→80) becomes the horizontal axis.
	p2 = plot(xlims=(-2, 82), ylims=(-2, 122), aspect_ratio=:equal,
		legend=false, title="Vertical Shot Map",
		background_color="#22312b", ticks=false, framestyle=:none)

	# Rebuild pitch lines in vertical orientation
	lw = 1.5; lc = "#efefef"
	plot!(p2, [0, 80, 80, 0, 0], [0, 0, 120, 120, 0], color=lc, lw=lw)
	plot!(p2, [0, 80], [60, 60], color=lc, lw=lw)
	θ = range(0, 2π; length=100)
	plot!(p2, 40 .+ 9.15 .* cos.(θ), 60 .+ 9.15 .* sin.(θ), color=lc, lw=lw)
	scatter!(p2, [40], [60], color=lc, markersize=3)
	# Penalty areas
	plot!(p2, [18, 18, 62, 62, 18], [0, 18, 18, 0, 0], color=lc, lw=lw)
	plot!(p2, [18, 18, 62, 62, 18], [120, 102, 102, 120, 120], color=lc, lw=lw)
	plot!(p2, [30, 30, 50, 50, 30], [0, 6, 6, 0, 0], color=lc, lw=lw)
	plot!(p2, [30, 30, 50, 50, 30], [120, 114, 114, 120, 120], color=lc, lw=lw)
	scatter!(p2, [40, 40], [12, 108], color=lc, markersize=4)
	# Corner arcs
	for (cx, cy, a0) in [(0, 0, 0), (80, 0, π/2), (80, 120, π), (0, 120, 3π/2)]
		θc = range(a0, a0 + π/2; length=20)
		plot!(p2, cx .+ 1 .* cos.(θc), cy .+ 1 .* sin.(θc), color=lc, lw=lw)
	end

	# Plot shots with swapped coordinates
	scatter!(p2, misses.y, misses.x,
		color=:red, markersize=5, alpha=0.5, label="Miss")
	scatter!(p2, goals.y, goals.x,
		color=:lime, markersize=10, label="Goal",
		markerstrokecolor=:white, markerstrokewidth=1.5)

	p2
end

# ╔═╡ e5f6a7b8-0021-6c4d-1f3e-0a1b2c3d4e5f
md"""
The vertical layout is compact, fits well in narrow columns, and can be
more intuitive for television audiences used to the camera angle behind
one goal.
"""

# ╔═╡ e5f6a7b8-0022-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Recap

In this notebook we:

1. Built a reusable `draw_pitch()` function for 120×80 StatsBomb coordinates
2. Created shot maps distinguishing goals from misses
3. Sized shot markers by xG to show chance quality
4. Built pass heatmaps with both histogram2d and 2D histogram
5. Created vertical (portrait) pitch layouts

## Julia vs mplsoccer

The Python `mplsoccer` library provides these visualizations out of the box.
In Julia, we build them from scratch with Plots.jl — more code, but total
control over every line, colour, and marker.  The `draw_pitch()` function
above can be reused across all your soccer notebooks.

## Next steps

- Passing networks (next notebook)
- Multi-panel figures comparing teams or time periods
- Animated sequences showing match progression
"""

# ╔═╡ e5f6a7b8-0023-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Exercises

1. **Defensive actions map** — plot tackles (Duel), interceptions, and
   clearances on the pitch with different colours.
2. **Comparison shot maps** — create side-by-side pitch plots showing shots
   from both teams in the match.
3. **Pass end locations** — create a heatmap of where passes *ended*, not
   where they started.  Use the `pass.end_location` array.
4. **Custom styling** — modify `draw_pitch()` to use your own colour scheme
   (try white pitch with black lines, or team colours).
5. **Time-split heatmaps** — create three pass heatmaps for 0–30′, 30–60′,
   and 60–90′ to see how passing patterns evolve.
"""

# ╔═╡ e5f6a7b8-0024-6c4d-1f3e-0a1b2c3d4e5f
md"""
## Solutions
"""

# ╔═╡ e5f6a7b8-0025-6c4d-1f3e-0a1b2c3d4e5f
md"""
### Exercise 1 — Defensive actions map

Plot tackles, interceptions, and clearances with distinct colours.
"""

# ╔═╡ e5f6a7b8-0026-6c4d-1f3e-0a1b2c3d4e5f
let
	def_types = ["Duel", "Interception", "Clearance"]
	def_events = subset(events, "type.name" => ByRow(t -> t in def_types))
	def_events.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(def_events.location)
	def_events.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(def_events.location)
	clean = dropmissing(def_events, [:x, :y])

	p = draw_pitch(title="Defensive Actions — Duel / Interception / Clearance")

	colors = Dict("Duel" => :red, "Interception" => :gold, "Clearance" => :cyan)
	for (typ, col) in colors
		sub = subset(clean, "type.name" => ByRow(==(typ)))
		scatter!(p, sub.x, sub.y, color=col, markersize=6, alpha=0.7, label=typ)
	end
	p
end

# ╔═╡ e5f6a7b8-0027-6c4d-1f3e-0a1b2c3d4e5f
md"""
### Exercise 2 — Side-by-side shot maps

Compare shot locations for both teams in the match.
"""

# ╔═╡ e5f6a7b8-0028-6c4d-1f3e-0a1b2c3d4e5f
let
	shots = subset(events, "type.name" => ByRow(==("Shot")))
	shots.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(shots.location)
	shots.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(shots.location)
	clean = dropmissing(shots, [:x, :y])

	teams = unique(skipmissing(clean[!, "team.name"]))
	n_teams = length(teams)

	plots_list = []
	for team in teams
		sub = subset(clean, "team.name" => ByRow(==(team)))
		goals = subset(sub, "shot.outcome.name" => ByRow(==("Goal")))
		misses = subset(sub, "shot.outcome.name" => ByRow(!=(missing)))
		misses = subset(misses, "shot.outcome.name" => ByRow(!=("Goal")))

		p_team = draw_pitch(title=team, bgcolor="#1a1a2e")
		scatter!(p_team, misses.x, misses.y, color=:red, markersize=4, alpha=0.5, label="Miss")
		scatter!(p_team, goals.x, goals.y, color=:lime, markersize=10, label="Goal",
			markerstrokecolor=:white, markerstrokewidth=1)
		push!(plots_list, p_team)
	end

	plot(plots_list..., layout=(1, n_teams), size=(1000, 450))
end

# ╔═╡ e5f6a7b8-0029-6c4d-1f3e-0a1b2c3d4e5f
md"""
### Exercise 3 — Pass end-location heatmap

Instead of pass starts, we plot where passes arrived.
"""

# ╔═╡ e5f6a7b8-0030-6c4d-1f3e-0a1b2c3d4e5f
let
	passes = subset(events,
		"type.name" => ByRow(==("Pass")),
		"team.name"  => ByRow(==("France Women's")))
	passes.ex = (endloc -> (ismissing(endloc) || endloc === nothing) ? missing : endloc[1]).(passes[!, "pass.end_location"])
	passes.ey = (endloc -> (ismissing(endloc) || endloc === nothing) ? missing : endloc[2]).(passes[!, "pass.end_location"])
	clean = dropmissing(passes, [:ex, :ey])

	p = draw_pitch(bgcolor="#1a1a2e", linecolor="#aaaaaa",
		title="Pass End Locations — France Women's")

	histogram2d!(p, clean.ex, clean.ey, gridsize=25, colorbar=true,
		colormap=:Reds, alpha=0.8, label="")

	p
end

# ╔═╡ e5f6a7b8-0031-6c4d-1f3e-0a1b2c3d4e5f
md"""
The heatmap shows where France's passes arrived.  Compare with the start
location heatmap: the end locations should be shifted toward the opponent's
goal.
"""

# ╔═╡ e5f6a7b8-0032-6c4d-1f3e-0a1b2c3d4e5f
md"""
### Exercise 4 — Custom styling

A light-themed pitch — white background, dark lines — for print or
presentation.
"""

# ╔═╡ e5f6a7b8-0033-6c4d-1f3e-0a1b2c3d4e5f
let
	# Draw a light-themed pitch by passing different colours
	p = draw_pitch(bgcolor="white", linecolor="#222222",
		title="Light-Theme Pitch")

	# Add a few random passes just to show contrast
	shots = subset(events, "type.name" => ByRow(==("Shot")))
	shots.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(shots.location)
	shots.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(shots.location)
	clean = dropmissing(shots, [:x, :y])
	goals = subset(clean, "shot.outcome.name" => ByRow(==("Goal")))

	scatter!(p, clean.x, clean.y, color=:darkred, markersize=5, alpha=0.6,
		label="Shot", markerstrokecolor=:white, markerstrokewidth=0.5)
	scatter!(p, goals.x, goals.y, color=:darkgreen, markersize=10,
		label="Goal", markerstrokecolor=:white, markerstrokewidth=1.5)

	p
end

# ╔═╡ e5f6a7b8-0034-6c4d-1f3e-0a1b2c3d4e5f
md"""
### Exercise 5 — Time-split heatmaps

Three panels showing pass density in each third of the match.
"""

# ╔═╡ e5f6a7b8-0035-6c4d-1f3e-0a1b2c3d4e5f
let
	passes = subset(events,
		"type.name" => ByRow(==("Pass")),
		"team.name"  => ByRow(==("France Women's")))
	passes.x = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[1]).(passes.location)
	passes.y = (loc -> (ismissing(loc) || loc === nothing) ? missing : loc[2]).(passes.location)
	clean = dropmissing(passes, [:x, :y])

	periods = [
		("0– 30′",  0, 30),
		("30– 60′", 30, 60),
		("60– 90′", 60, 90),
	]

	plots_list = []
	for (label, lo, hi) in periods
		sub = subset(clean, :minute => ByRow(m -> lo <= m < hi))
		p_period = draw_pitch(bgcolor="#1a1a2e",
			title="$label — $(nrow(sub)) passes")
		if nrow(sub) > 0
			histogram2d!(p_period, sub.x, sub.y, bins=(30, 20),
				colormap=:YlOrRd, alpha=0.7, label="", colorbar=false)
		end
		push!(plots_list, p_period)
	end

	plot(plots_list..., layout=(1, 3), size=(1500, 400))
end

# ╔═╡ e5f6a7b8-0036-6c4d-1f3e-0a1b2c3d4e5f
md"""
Passing patterns often evolve through the match: early probing, mid-game
consolidation, and late urgency when chasing a result.
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
# ╟─e5f6a7b8-0001-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0002-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0003-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0004-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0005-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0006-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0007-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0008-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0009-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0010-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0011-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0012-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0013-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0014-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0015-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0016-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0017-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0018-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0019-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0020-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0021-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0022-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0023-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0024-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0025-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0026-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0027-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0028-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0029-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0030-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0031-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0032-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0033-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0034-6c4d-1f3e-0a1b2c3d4e5f
# ╠═e5f6a7b8-0035-6c4d-1f3e-0a1b2c3d4e5f
# ╟─e5f6a7b8-0036-6c4d-1f3e-0a1b2c3d4e5f
# ╠═00000000-0000-0000-0000-000000000001
