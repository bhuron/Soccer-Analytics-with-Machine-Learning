### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 8a1b2c3d-0003-4e1f-9b1a-7e6f5a4b3c2e
begin
	using JSON3
	using DataFrames
	using Statistics
	using Chain: @chain
end

# ╔═╡ 8a1b2c3d-0001-4e1f-9b1a-7e6f5a4b3c2d
md"""
# Loading StatsBomb Soccer Data

**Chapter 3 · Exploratory Data Analysis in Soccer**

## What you'll learn

- Grab StatsBomb's open data straight from GitHub
- Parse nested match JSON into a tidy `DataFrame`
- Run first-pass data quality checks
- Navigate the structure of the dataset
"""

# ╔═╡ 8a1b2c3d-0002-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Imports & setup
"""

# ╔═╡ 8a1b2c3d-0004-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Where the data lives

[StatsBomb](https://statsbomb.com/) is one of the heavyweights in soccer analytics.
Their [open-data](https://github.com/statsbomb/open-data) repo on GitHub is a goldmine —
every pass, shot, dribble, and tackle across several competitions, all logged at
the event level.

Grab it once and you're set:

```bash
git clone https://github.com/statsbomb/open-data.git
```

The repo unpacks into `open-data/`.  Inside, `data/` holds the actual JSON files,
organised by competition and season.  For the 2019 Women's World Cup:

```
open-data/data/matches/72/30.json
```

- `72`  — competition ID (FIFA Women's World Cup)
- `30`  — season ID (2019)

Let's point our code at that file.  (If you cloned somewhere else, tweak the path.)
"""

# ╔═╡ 8a1b2c3d-0005-4e1f-9b1a-7e6f5a4b3c2e
begin
	# Path relative to the notebook location (notebooks/chapter-3/)
	const DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")
	const MATCH_FILE = joinpath(DATA_DIR, "matches", "72", "30.json")
	
	# Verify the file exists so we fail early with a clear message
	isfile(MATCH_FILE) || error("Match file not found at $MATCH_FILE. Did you clone https://github.com/statsbomb/open-data ?")
	
	md"""
	Data directory: `$DATA_DIR`
	"""
end

# ╔═╡ 8a1b2c3d-0006-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Loading the match list

StatsBomb stores its matches as a JSON array of nested objects.
We read the whole thing into memory, then flatten the nested keys
(e.g. `home_team.home_team_name`) into flat column names — the Julia
equivalent of pandas' `json_normalize`.
"""

# ╔═╡ 8a1b2c3d-0007-4e1f-9b1a-7e6f5a4b3c2e
"""
    flatten_dict(d::AbstractDict; prefix="")

Recursively flatten a nested dictionary, joining keys with `"."`.
A leaf `"France"` nested under `:home_team => :home_team_name` becomes
`"home_team.home_team_name" => "France"`.
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

# ╔═╡ 8a1b2c3d-0008-4e1f-9b1a-7e6f5a4b3c2e
begin
	# Read and flatten.
	raw = JSON3.read(read(MATCH_FILE, String))

	# Flatten each match; then collect *all* keys across matches so
	# that heterogeneous fields (key present in match A but not B)
	# don't trip up the DataFrame constructor.
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)

	rows = [let row = Dict{String,Any}()
	    for k in all_keys
	        row[k] = get(d, k, missing)
	    end
	    row
	end for d in dicts]

	matches_df = DataFrame(rows)

	# Quick sanity check
	n_matches = nrow(matches_df)
	n_cols = ncol(matches_df)
	md"""
	Loaded **$n_matches matches** across **$n_cols columns**.
	"""
end

# ╔═╡ 8a1b2c3d-0009-4e1f-9b1a-7e6f5a4b3c2e
# Peek at the first few rows and columns
first(matches_df, 5)

# ╔═╡ 8a1b2c3d-0010-4e1f-9b1a-7e6f5a4b3c2e
md"""
## First look

Before diving in, let's check what we're working with: column names,
types, and how much is actually populated.
"""

# ╔═╡ 8a1b2c3d-0011-4e1f-9b1a-7e6f5a4b3c2e
md"""
### Column names, types & missingness

`DataFrame`'s `describe` gives a compact summary of every column.
"""

# ╔═╡ 8a1b2c3d-0012-4e1f-9b1a-7e6f5a4b3c2e
describe(matches_df)

# ╔═╡ 8a1b2c3d-0013-4e1f-9b1a-7e6f5a4b3c2e
md"""
### Missing values per column

A quick tally so we know where the gaps are.
"""

# ╔═╡ 8a1b2c3d-0014-4e1f-9b1a-7e6f5a4b3c2e
begin
    # Tally missing values per column
    missing_counts = let
        counts = [count(ismissing, matches_df[!, c]) for c in names(matches_df)]
        sort(DataFrame(column=names(matches_df), n_missing=counts), :n_missing, rev=true)
    end
    
    # Show only columns with at least one missing value
    filter(:n_missing => >(0), missing_counts)
end

# ╔═╡ 8a1b2c3d-0015-4e1f-9b1a-7e6f5a4b3c2e
md"""
### Numeric summaries

A stats-table view of the numeric columns — just like `df.describe()` in pandas.
"""

# ╔═╡ 8a1b2c3d-0016-4e1f-9b1a-7e6f5a4b3c2e
begin
    # Collect columns whose element type is numeric (includes Union{T, Missing} where T <: Number)
    num_cols = [n for n in names(matches_df) if eltype(matches_df[!, n]) <: Union{Number,Missing}]
    num_stats = let
        df_num = matches_df[!, num_cols]
        stats = DataFrame(
            column   = num_cols,
            mean     = [mean(skipmissing(df_num[!, c])) for c in num_cols],
            std      = [std(skipmissing(df_num[!, c]))  for c in num_cols],
            min_val  = [minimum(skipmissing(df_num[!, c])) for c in num_cols],
            max_val  = [maximum(skipmissing(df_num[!, c])) for c in num_cols],
            n_missing= [count(ismissing, df_num[!, c]) for c in num_cols],
        )
        sort(stats, :n_missing, rev=true)
    end
    num_stats
end

# ╔═╡ 8a1b2c3d-0017-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Key columns

Let's pull out the columns that matter most for analysis and have a look.
"""

# ╔═╡ 8a1b2c3d-0018-4e1f-9b1a-7e6f5a4b3c2e
@chain matches_df begin
    select(
        _,
        :match_id,
        :match_date,
        "home_team.home_team_name"    => :home_team,
        "away_team.away_team_name"    => :away_team,
        :home_score,
        :away_score,
        "competition_stage.name"      => :stage,
    )
    first(_, 10)
end

# ╔═╡ 8a1b2c3d-0019-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Tournament structure

Which stages does the competition go through?
"""

# ╔═╡ 8a1b2c3d-0020-4e1f-9b1a-7e6f5a4b3c2e
begin
    stage_col = "competition_stage.name"
    
    if stage_col in names(matches_df)
        stage_counts = @chain matches_df begin
            groupby(_, stage_col; sort=true)
            combine(nrow => :matches)
            rename!(_, stage_col => :stage)
        end
        stage_counts
    else
        md"""!!! warning "Column not found"
    `$(stage_col)` is missing. Available columns with *stage* in the name: $(filter(c -> occursin("stage", c), names(matches_df)))"""
    end
end

# ╔═╡ 8a1b2c3d-0021-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Recap

In this notebook we:

1. Cloned StatsBomb's open data from GitHub
2. Loaded the 2019 Women's World Cup match list into a `DataFrame`
3. Flattened nested JSON with a recursive helper (no pandas needed 😉)
4. Checked column types, missing values, and numeric summaries with `describe` and a quick stats table
5. Surfaced key columns and tournament stages

Already we can answer useful questions — how many matches, which teams played,
which knockout stages are present.

## Up next

The [next notebook](./03-exploring-match-data.ipynb) digs into match-level
numbers: goals, match duration, and competition-level stats.  We'll also
tackle event data — passes, shots, tackles — and lineups.
"""

# ╔═╡ 8a1b2c3d-0022-4e1f-9b1a-7e6f5a4b3c2e
md"""
## Exercises

1. **All the teams** — build a sorted list of unique teams that appeared in the tournament.
2. **Date span** — find the first and last match date.
3. **Goal fests** — which matches had 5+ total goals?
4. **Column audit** — list every column and mark which ones would be useful for match-level analysis vs. event-level analysis.
"""

# ╔═╡ 9b2c3d4e-0001-5f2a-8c2b-7e6f5a4b3c2d
md"""
## Solutions

Below are detailed answers to each exercise. Try solving them on your own first —
the learning happens in the struggle. Then come back here to compare notes.
"""

# ╔═╡ 9b2c3d4e-0002-5f2a-8c2b-7e6f5a4b3c2d
md"""
### Exercise 1 — All the teams

We need every unique team name, whether they played at home or away. The two
columns live at `home_team.home_team_name` and `away_team.away_team_name`.
Julia's `vcat` concatenates the two columns, `unique` discards duplicates,
and `sort` puts them in alphabetical order.
"""

# ╔═╡ 9b2c3d4e-0003-5f2a-8c2b-7e6f5a4b3c2d
let
    home = matches_df[!, "home_team.home_team_name"]
    away = matches_df[!, "away_team.away_team_name"]
    sort(unique(vcat(home, away)))
end

# ╔═╡ 9b2c3d4e-0004-5f2a-8c2b-7e6f5a4b3c2d
md"""
24 teams, as you'd expect for a World Cup. Notice how `vcat(home, away)`
stacks the two columns on top of each other — we don't care about home/away
status here, only about who participated. `unique` then collapses the
duplicates (each team appears in both columns at least once).
"""

# ╔═╡ 9b2c3d4e-0005-5f2a-8c2b-7e6f5a4b3c2d
md"""
### Exercise 2 — Date span

The match dates are ISO 8601 strings (`"2019-07-03"`). Because the format is
`YYYY-MM-DD`, lexicographic order matches chronological order — we can sort
without parsing. `extrema` returns `(min, max)` in one shot.
"""

# ╔═╡ 9b2c3d4e-0006-5f2a-8c2b-7e6f5a4b3c2d
let
    dates = matches_df[!, :match_date]
    first_date, last_date = extrema(dates)
    md"""
    | | Date |
    |---|---|
    | **First match** | $(first_date) |
    | **Last match** | $(last_date) |
    """
end

# ╔═╡ 9b2c3d4e-0007-5f2a-8c2b-7e6f5a4b3c2d
md"""
A 4-day gap between the opening match on June 7 and the final on July 7 —
typical for a month-long tournament. If you needed to compute the duration in
days you'd reach for `Dates.Date`, but for a quick min/max, string comparison
does the job.
"""

# ╔═╡ 9b2c3d4e-0008-5f2a-8c2b-7e6f5a4b3c2d
md"""
### Exercise 3 — Goal fests

Matches where the total goals (home + away) reached 5 or more. `@chain` reads
left-to-right: filter → compute total → pick display columns → sort by most
exciting first.
"""

# ╔═╡ 9b2c3d4e-0009-5f2a-8c2b-7e6f5a4b3c2d
@chain matches_df begin
    transform(_, [:home_score, :away_score] => ByRow(+) => :total_goals)
    filter(:total_goals => >=(5), _)
    select(:match_date,
        "home_team.home_team_name" => :home,
        "away_team.away_team_name" => :away,
        :home_score, :away_score, :total_goals)
    sort(_, :total_goals, rev=true)
end

# ╔═╡ 9b2c3d4e-0010-5f2a-8c2b-7e6f5a4b3c2d
md"""
`ByRow(+)` tells DataFrames to apply `+` to each pair of values rather than
the whole column vectors at once (the array-level `+` would work too here,
but `ByRow` makes the intent explicit). The standout is the USA–Thailand
group-stage match at 13–0, which remains the largest margin of victory in
Women's World Cup history.
"""

# ╔═╡ 9b2c3d4e-0011-5f2a-8c2b-7e6f5a4b3c2d
md"""
### Exercise 4 — Column audit

Not all 42 columns carry the same analytical weight. We group them by
**purpose** so you can quickly decide which ones matter for a given question:

| Category | Examples | Use when... |
|---|---|---|
| **match result** | `home_score`, `away_score` | predicting outcomes |
| **team info** | `home_team.*`, `away_team.*` | team-level comparisons |
| **match metadata** | `match_date`, `kick_off`, `competition_stage.name` | temporal or competition context |
| **venue** | `stadium.*` | home-advantage analysis |
| **officials** | `referee.*` | referee bias checks |
| **personnel** | `*.managers` | coaching analysis |
| **data version** | `metadata.*`, `last_updated*` | data provenance (ignore for modeling) |
"""

# ╔═╡ 9b2c3d4e-0012-5f2a-8c2b-7e6f5a4b3c2d
let
    classify(c::String) = if any(startswith(c, p) for p in ["home_team", "away_team"])
            "team info"
        elseif any(startswith(c, p) for p in ["home_score", "away_score"])
            "match result"
        elseif any(startswith(c, p) for p in ["competition", "season", "match_date",
                "match_week", "match_status", "kick_off", "last_updated"])
            "match metadata"
        elseif startswith(c, "stadium")
            "venue"
        elseif startswith(c, "referee")
            "officials"
        elseif endswith(c, ".managers") || endswith(c, "managers")
            "personnel"
        elseif startswith(c, "metadata")
            "data version"
        else
            "identifier / other"
        end

    cols = names(matches_df)
    sort(DataFrame(column=cols, category=classify.(cols)), [:category, :column])
end

# ╔═╡ 9b2c3d4e-0013-5f2a-8c2b-7e6f5a4b3c2d
md"""
**Key insight**: columns tagged `data version` or `identifier / other` are
book-keeping fields — they won't help a model learn about soccer. The `team
info` and `match result` columns are your bread and butter for analysis.
`personnel` columns (manager data) are JSON blobs that need further unpacking
before they're useful.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Chain = "~1.0.0"
DataFrames = "~1.8.2"
JSON3 = "~1.14.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "e80b6e3fce0309b31a7e04c4402f64bc6dd882bb"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Chain]]
git-tree-sha1 = "765487f32aeece2cf28aa7038e29c31060cb5a69"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "468dbe2b510c876dc091b2c74ed52c7c34f48b9b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.5"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─8a1b2c3d-0001-4e1f-9b1a-7e6f5a4b3c2d
# ╟─8a1b2c3d-0002-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0003-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0004-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0005-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0006-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0007-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0008-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0009-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0010-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0011-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0012-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0013-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0014-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0015-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0016-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0017-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0018-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0019-4e1f-9b1a-7e6f5a4b3c2e
# ╠═8a1b2c3d-0020-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0021-4e1f-9b1a-7e6f5a4b3c2e
# ╟─8a1b2c3d-0022-4e1f-9b1a-7e6f5a4b3c2e
# ╟─9b2c3d4e-0001-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0002-5f2a-8c2b-7e6f5a4b3c2d
# ╠═9b2c3d4e-0003-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0004-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0005-5f2a-8c2b-7e6f5a4b3c2d
# ╠═9b2c3d4e-0006-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0007-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0008-5f2a-8c2b-7e6f5a4b3c2d
# ╠═9b2c3d4e-0009-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0010-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0011-5f2a-8c2b-7e6f5a4b3c2d
# ╠═9b2c3d4e-0012-5f2a-8c2b-7e6f5a4b3c2d
# ╟─9b2c3d4e-0013-5f2a-8c2b-7e6f5a4b3c2d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
