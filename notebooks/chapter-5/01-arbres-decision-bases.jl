### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ d2e3f4a5-0003-3b1c-9e0f-5a6b7c8d9e0f
begin
	using JSON3
	using DataFrames
	using Statistics
	using Random
	using DecisionTree
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ d2e3f4a5-0001-3b1c-9e0f-5a6b7c8d9e0f
md"""
# Arbres de décision — la logique du football en règles

**Chapitre 5 · Méthodes avancées de classification**

## Ce que vous allez apprendre

- Comprendre comment fonctionnent les arbres de décision
- Construire un arbre pour prédire la réussite d'une passe
- Visualiser et interpréter la structure de l'arbre
- Comprendre les règles de décision
- Connaître les avantages et limites des arbres
"""

# ╔═╡ d2e3f4a5-0002-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Imports et configuration
"""

# ╔═╡ d2e3f4a5-0004-3b1c-9e0f-5a6b7c8d9e0f
md"""
## L'intuition — apprendre des règles « si… alors… »

Un **arbre de décision**, c'est comme jouer aux 20 questions avec vos
données.  À chaque nœud, on pose une question simple ; la réponse
détermine la branche suivante.

**Exemple pour une passe :**
1. La distance est-elle < 30 m ? → Si OUI, question 2 ; si NON → « Ratée »
2. Le point de départ est-il dans la moitié adverse ? → Si OUI, question 3
3. La passe est-elle au sol ? → Si OUI → « Réussie » ; si NON → « Ratée »

Cela crée un modèle **transparent et lisible par un humain**.
"""

# ╔═╡ d2e3f4a5-0005-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Charger les données — passes de la finale de la LDC 2019
"""

# ╔═╡ d2e3f4a5-0006-3b1c-9e0f-5a6b7c8d9e0f
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

	raw = JSON3.read(read(joinpath(DATA_DIR, "events", "22912.json"), String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)
	rows = [let row = Dict{String,Any}()
		for k in all_keys; row[k] = get(d, k, missing); end; row
	end for d in dicts]
	events_dt = DataFrame(rows)

	# Filter passes
	passes_dt = subset(events_dt, "type.name" => ByRow(==("Pass")); skipmissing=true)
	passes_dt.completed = ismissing.(passes_dt[!, "pass.outcome.name"]) .|> Int

	# Extract start coordinates
	passes_dt.start_x = [(ismissing(loc) || loc === nothing) ? missing : loc[1] for loc in passes_dt.location]
	passes_dt.start_y = [(ismissing(loc) || loc === nothing) ? missing : loc[2] for loc in passes_dt.location]

	# Compute pass length from end_location
	passes_dt.pass_length = Vector{Union{Missing,Float64}}(missing, nrow(passes_dt))
	passes_dt.end_x = Vector{Union{Missing,Float64}}(missing, nrow(passes_dt))
	passes_dt.end_y = Vector{Union{Missing,Float64}}(missing, nrow(passes_dt))
	for i in 1:nrow(passes_dt)
		el = passes_dt[i, "pass.end_location"]
		if !ismissing(el) && el !== nothing
			passes_dt.end_x[i] = el[1]
			passes_dt.end_y[i] = el[2]
			if !ismissing(passes_dt.start_x[i])
				passes_dt.pass_length[i] = sqrt((el[1] - passes_dt.start_x[i])^2 + (el[2] - passes_dt.start_y[i])^2)
			end
		end
	end

	global passes_dt
	md"""**$(nrow(passes_dt)) passes** chargées — taux de réussite : **$(round(100 * mean(skipmissing(passes_dt.completed)); digits=1)) %**"""
end

# ╔═╡ d2e3f4a5-0007-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Préparer les données d'entraînement
"""

# ╔═╡ d2e3f4a5-0008-3b1c-9e0f-5a6b7c8d9e0f
let
	clean = dropmissing(passes_dt, [:start_x, :start_y, :pass_length, :completed])
	features = [:start_x, :start_y, :pass_length]
	X_mat = Matrix(clean[!, features])
	y_vec = Vector(clean[!, :completed])

	n_dt = nrow(clean)
	train_ix = shuffle(1:n_dt)[1:Int(round(0.8 * n_dt))]
	test_ix  = setdiff(1:n_dt, train_ix)

	global X_train_dt = X_mat[train_ix, :]
	global y_train_dt = y_vec[train_ix]
	global X_test_dt  = X_mat[test_ix, :]
	global y_test_dt  = y_vec[test_ix]
	global feature_names_dt = string.(features)

	md"""
	- Entraînement : **$(length(train_ix)) passes**
	- Test : **$(length(test_ix)) passes**
	- Variables : start_x, start_y, pass_length

	**Distribution des classes (entraînement) :**
	- Réussies : $(sum(y_train_dt)) ($(round(100 * mean(y_train_dt); digits=1)) %)
	- Ratées : $(length(y_train_dt) - sum(y_train_dt))
	"""
end

# ╔═╡ d2e3f4a5-0009-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Entraîner l'arbre de décision

Avec `DecisionTree.jl` : `max_depth=3` pour un arbre interprétable.
"""

# ╔═╡ d2e3f4a5-0010-3b1c-9e0f-5a6b7c8d9e0f
let
	# DecisionTree.jl expects X as (features × samples)
	model = DecisionTreeClassifier(max_depth=3)
	fit!(model, X_train_dt', y_train_dt)
	global tree_model = model

	y_pred = predict(model, X_test_dt')
	acc = sum(y_pred .== y_test_dt) / length(y_test_dt)

	md"""
	**Arbre entraîné (max_depth=3) :**

	Accuracy sur le test : **$(round(acc * 100; digits=1)) %**

	"""
end

# ╔═╡ d2e3f4a5-0011-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Interpréter l'arbre

Prenons un exemple de règle apprise :

- Si `pass_length ≤ 30 m` (passe courte)
  - ET `start_x > 110` (dans le tiers offensif)
  - ALORS prédire **Réussie** (haute probabilité)

**Intuition footballistique :** les passes courtes dans le dernier tiers
sont très souvent réussies.  C'est cohérent avec la sagesse tactique
— conserver le ballon dans les zones dangereuses.
"""

# ╔═╡ d2e3f4a5-0012-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Importance des variables

Quelles variables comptent le plus pour les décisions de l'arbre ?
"""

# ╔═╡ d2e3f4a5-0013-3b1c-9e0f-5a6b7c8d9e0f
let
	importances = feature_importances(tree_model)
	imp_df = sort(DataFrame(
		variable   = feature_names_dt,
		importance = round.(importances; digits=3)),
		:importance, rev=true)

	bar(imp_df.variable, imp_df.importance,
		legend=false, color=:steelblue, alpha=0.7,
		title="Importance des variables dans l'arbre",
		xrotation=45, ylabel="Importance")

	imp_df
end

# ╔═╡ d2e3f4a5-0014-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Avantages des arbres de décision

| Avantage | Description |
|---|---|
| **Transparence** | Modèle boîte blanche — chaque prédiction est explicable |
| **Pas de standardisation** | Fonctionne avec des valeurs brutes |
| **Relations non linéaires** | Capture des interactions complexes |
| **Visualisable** | On peut littéralement lire la logique de décision |
| **Données mixtes** | Accepte variables numériques et catégorielles |
"""

# ╔═╡ d2e3f4a5-0015-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Inconvénients des arbres de décision

| Inconvénient | Description |
|---|---|
| **Variance élevée** | Un petit changement dans les données → arbre complètement différent |
| **Surapprentissage** | Les arbres profonds mémorisent les données d'entraînement |
| **Puissance limitée** | Un arbre seul est souvent moins précis qu'un ensemble |
| **Algorithme glouton** | Optimise localement, peut manquer l'optimum global |
"""

# ╔═╡ d2e3f4a5-0016-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Compris comment les arbres de décision fonctionnent (règles si-alors)
2. Construit un arbre pour prédire la réussite des passes
## Points clés

- Les arbres de décision sont **hautement interprétables**
- Idéal pour l'**analyse exploratoire** et la présentation à des non-experts
- Un arbre seul est **instable** et sujet au surapprentissage
- Nécessite un **réglage** (max_depth, min_samples_split) pour éviter le surapprentissage

## Prochaine étape

Dans le prochain notebook, nous apprendrons à **ajuster les arbres de
décision** pour éviter le surapprentissage et améliorer les performances !
"""

# ╔═╡ d2e3f4a5-0017-3b1c-9e0f-5a6b7c8d9e0f
md"""
## Exercices

1. **Tester différentes profondeurs** — entraîner des arbres avec
   max_depth de 1 à 10 et comparer l'accuracy.
2. **Ajouter plus de variables** — inclure la hauteur de passe
   (`pass.height.name`), la partie du corps (`pass.body_part.name`).
3. **Prédire autre chose** — prédire les tirs (but/pas but) au lieu
   des passes.
4. **Visualiser les frontières** — tracer les régions de décision
   dans l'espace 2D (start_x × pass_length).
5. **Extraire les règles** — écrire le code pour extraire toutes les
   règles de l'arbre sous forme de texte lisible.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
DecisionTree = "0.12"
JSON3 = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─d2e3f4a5-0001-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0002-3b1c-9e0f-5a6b7c8d9e0f
# ╠═d2e3f4a5-0003-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0004-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0005-3b1c-9e0f-5a6b7c8d9e0f
# ╠═d2e3f4a5-0006-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0007-3b1c-9e0f-5a6b7c8d9e0f
# ╠═d2e3f4a5-0008-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0009-3b1c-9e0f-5a6b7c8d9e0f
# ╠═d2e3f4a5-0010-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0011-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0012-3b1c-9e0f-5a6b7c8d9e0f
# ╠═d2e3f4a5-0013-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0014-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0015-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0016-3b1c-9e0f-5a6b7c8d9e0f
# ╟─d2e3f4a5-0017-3b1c-9e0f-5a6b7c8d9e0f
# ╠═00000000-0000-0000-0000-000000000001
