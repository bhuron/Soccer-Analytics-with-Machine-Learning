### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ e7f8a9b0-0003-8c6d-4f5a-0b1c2d3e4f5a
begin
	using JSON3
	using DataFrames
	using Statistics
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=10)
end

# ╔═╡ e7f8a9b0-0001-8c6d-4f5a-0b1c2d3e4f5a
md"""
# Introduction à la classification dans le football

**Chapitre 4 · Prédire les résultats de matchs avec la classification**

## Ce que vous allez apprendre

- Comprendre ce qu'est la classification et en quoi elle diffère de la régression
- Distinguer les problèmes de classification binaire et multi-classe
- Identifier les problèmes de classification dans l'analyse du football
- Comprendre le workflow de base d'un projet de classification
"""

# ╔═╡ e7f8a9b0-0002-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Imports et configuration
"""

# ╔═╡ e7f8a9b0-0004-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Qu'est-ce que la classification ?

La classification consiste à **ranger** des choses dans des catégories.
Au lieu de prédire un nombre continu (comme la régression), on prédit
une **étiquette discrète** ou une **classe**.

Pensez à ces questions footballistiques :
- Ce tir sera-t-il un **but** ou **non** ?
- Cette passe sera-t-elle **réussie** ou **interceptée** ?
- Le match se terminera-t-il par une **victoire**, un **nul** ou une **défaite** ?

Ce sont tous des problèmes de classification !
"""

# ╔═╡ e7f8a9b0-0005-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Classification binaire vs multi-classe

### Classification binaire

Deux résultats possibles. Exemples :

1. **But ou pas ?** — base des modèles d'Expected Goals (xG)
2. **Passe réussie ou non ?** — évaluation de la qualité de passe
3. **Tacle réussi ou raté ?** — évaluation défensive

### Classification multi-classe

Trois catégories ou plus. Exemples :

1. **Résultat du match :** Victoire, Nul, Défaite
2. **Type de passe :** Passe courte, longue balle, centre
3. **Type de tir :** Jeu ouvert, coup franc, penalty

On commence par la classification binaire, plus simple à comprendre.
"""

# ╔═╡ e7f8a9b0-0006-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Exemple réel — les tirs d'un match

Chargeons des données réelles de la finale de la Ligue des Champions 2019.
"""

# ╔═╡ e7f8a9b0-0007-8c6d-4f5a-0b1c2d3e4f5a
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

	# Load Champions League final events
	event_file = joinpath(DATA_DIR, "events", "22912.json")
	raw = JSON3.read(read(event_file, String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)
	rows = [let row = Dict{String,Any}()
		for k in all_keys; row[k] = get(d, k, missing); end; row
	end for d in dicts]

	global events_cl = DataFrame(rows)
	md"""**$(nrow(events_cl)) événements** chargés depuis la finale de la Ligue des Champions 2019."""
end

# ╔═╡ e7f8a9b0-0008-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Analyser les tirs du match
"""

# ╔═╡ e7f8a9b0-0009-8c6d-4f5a-0b1c2d3e4f5a
let
	shots = subset(events_cl, "type.name" => ByRow(==("Shot")); skipmissing=true)
	shots.is_goal = isequal.(shots[!, "shot.outcome.name"], "Goal")

	n_shots = nrow(shots)
	n_goals = sum(shots.is_goal)
	n_nongoals = n_shots - n_goals

	outcome_counts = combine(groupby(shots, "shot.outcome.name"), nrow => :count)
	sort!(outcome_counts, :count, rev=true)

	p1 = bar(["Pas but", "But"], [n_nongoals, n_goals],
		legend=false, color=[:coral, :lime], alpha=0.7,
		title="Tirs : classification binaire", ylabel="Nombre")

	outcome_counts
end

# ╔═╡ e7f8a9b0-0010-8c6d-4f5a-0b1c2d3e4f5a
md"""
**Ce que l'on observe :**
- La plupart des tirs ne sont **pas des buts** (arrêtés, bloqués, hors cadre)
- Seuls quelques tirs aboutissent à un **but**
- C'est un problème de **classification binaire** : But vs Pas but
- Les classes sont **déséquilibrées** (plus de non-buts que de buts)
"""

# ╔═╡ e7f8a9b0-0011-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Le workflow de la classification

Chaque projet de classification suit ces étapes :

1. **Définir le problème** — que cherchons-nous à prédire ?
2. **Collecter les données** — rassembler les variables et résultats pertinents
3. **Préparer les données** — nettoyer, transformer, créer des variables
4. **Choisir l'algorithme** — sélectionner le classifieur approprié
5. **Entraîner le modèle** — apprendre les motifs des données d'entraînement
6. **Évaluer la performance** — tester sur des données non vues
7. **Interpréter les résultats** — comprendre ce que le modèle a appris

Nous suivrons ce workflow tout au long du chapitre 4.
"""

# ╔═╡ e7f8a9b0-0012-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Récapitulatif

Dans ce notebook, nous avons appris :

1. **La classification prédit des catégories**, pas des nombres
2. **La classification binaire** a deux résultats (But/Pas but)
3. **La classification multi-classe** a trois résultats ou plus (Victoire/Nul/Défaite)
4. **Les vraies données de football** ont souvent des classes déséquilibrées
5. **Le workflow de classification** guide nos projets

## Prochaine étape

Dans le prochain notebook, nous construirons notre premier classifieur :
un modèle de **régression logistique** pour prédire les Expected Goals (xG) !
"""

# ╔═╡ e7f8a9b0-0013-8c6d-4f5a-0b1c2d3e4f5a
md"""
## Exercices

1. **Identifier des problèmes de classification** — lister 5 autres problèmes
   de classification binaire dans le football.
2. **Exemples multi-classe** — lister 3 problèmes de classification multi-classe.
3. **Charger d'autres données** — charger un match différent et examiner les
   résultats des tirs.
4. **Équilibre des classes** — calculer le pourcentage de buts dans différentes
   compétitions.
5. **Brainstorming de variables** — quelles variables pourraient aider à
   prédire si un tir sera un but ?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
JSON3 = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─e7f8a9b0-0001-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0002-8c6d-4f5a-0b1c2d3e4f5a
# ╠═e7f8a9b0-0003-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0004-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0005-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0006-8c6d-4f5a-0b1c2d3e4f5a
# ╠═e7f8a9b0-0007-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0008-8c6d-4f5a-0b1c2d3e4f5a
# ╠═e7f8a9b0-0009-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0010-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0011-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0012-8c6d-4f5a-0b1c2d3e4f5a
# ╟─e7f8a9b0-0013-8c6d-4f5a-0b1c2d3e4f5a
# ╠═00000000-0000-0000-0000-000000000001
