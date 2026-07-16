### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ f8a9b0c1-0003-9d7e-5a6b-1c2d3e4f5a6b
begin
	using JSON3
	using DataFrames
	using Statistics
	using LinearAlgebra
	using GLM
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=10)
end

# ╔═╡ f8a9b0c1-0001-9d7e-5a6b-1c2d3e4f5a6b
md"""
# Régression logistique et Expected Goals (xG)

**Chapitre 4 · Prédire les résultats de matchs avec la classification**

## Ce que vous allez apprendre

- Comprendre comment fonctionne la régression logistique
- Construire un modèle d'Expected Goals (xG) avec la régression logistique
- Créer des variables à partir des données d'événements
- Interpréter les coefficients du modèle
- Faire des prédictions avec votre modèle entraîné
"""

# ╔═╡ f8a9b0c1-0002-9d7e-5a6b-1c2d3e4f5a6b
md"""
## Imports et configuration
"""

# ╔═╡ f8a9b0c1-0004-9d7e-5a6b-1c2d3e4f5a6b
md"""
## Qu'est-ce que la régression logistique ?

La régression logistique est un **algorithme de classification** qui prédit
des **probabilités** entre 0 et 1.

### La fonction sigmoïde

La clé de la régression logistique est la **fonction sigmoïde**, qui
« compresse » n'importe quel nombre en une probabilité :

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Où :
- $z$ = un « score » calculé à partir des variables
- $\sigma(z)$ = probabilité entre 0 et 1

**Propriétés :**
- Score très négatif → probabilité proche de 0
- Score très positif → probabilité proche de 1
- Score de 0 → probabilité de 0,5
"""

# ╔═╡ f8a9b0c1-0005-9d7e-5a6b-1c2d3e4f5a6b
let
	z = range(-10, 10; length=100)
	sigmoid(z) = 1 / (1 + exp(-z))

	plot(z, sigmoid.(z), linewidth=2, legend=false, color=:steelblue,
		title="La fonction sigmoïde", xlabel="Score (z)", ylabel="Probabilité")
	hline!([0.5], color=:red, linestyle=:dash, alpha=0.6, label="Seuil = 0,5")
	vline!([0], color=:green, linestyle=:dash, alpha=0.6, label="Score = 0")
end

# ╔═╡ f8a9b0c1-0006-9d7e-5a6b-1c2d3e4f5a6b
md"""
## Construire un modèle d'Expected Goals (xG)

Objectif : prédire la probabilité qu'un tir se transforme en but.

### Étape 1 — Charger les données

Finale de la Ligue des Champions 2019 entre Liverpool et Tottenham.
"""

# ╔═╡ f8a9b0c1-0007-9d7e-5a6b-1c2d3e4f5a6b
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

	event_file = joinpath(DATA_DIR, "events", "22912.json")
	raw = JSON3.read(read(event_file, String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)
	rows = [let row = Dict{String,Any}()
		for k in all_keys; row[k] = get(d, k, missing); end; row
	end for d in dicts]

	global events_xg = DataFrame(rows)
	md"""**$(nrow(events_xg)) événements** chargés depuis la finale Liverpool vs Tottenham."""
end

# ╔═╡ f8a9b0c1-0008-9d7e-5a6b-1c2d3e4f5a6b
md"""
### Étape 2 — Feature engineering

Créons deux variables clés :
1. **Distance au but** — à quelle distance le tir a-t-il été effectué ?
2. **Angle au but** — quel angle le tireur avait-il par rapport au but ?

Dans StatsBomb, le but est situé aux coordonnées (120, 40) et mesure
7,32 m de large.
"""

# ╔═╡ f8a9b0c1-0009-9d7e-5a6b-1c2d3e4f5a6b
let
	shots = subset(events_xg, "type.name" => ByRow(==("Shot")); skipmissing=true)
	shots.goal = isequal.(shots[!, "shot.outcome.name"], "Goal") .|> Int

	# Extract x, y from location
	shots.x = [(ismissing(loc) || loc === nothing) ? missing : loc[1] for loc in shots.location]
	shots.y = [(ismissing(loc) || loc === nothing) ? missing : loc[2] for loc in shots.location]
	clean = dropmissing(shots, [:x, :y])

	# Distance and angle
	clean.distance = sqrt.((120 .- clean.x).^2 .+ (40 .- clean.y).^2)
	goal_width = 7.32
	clean.angle = atand.(goal_width .* (120 .- clean.x) ./
		((120 .- clean.x).^2 .+ (40 .- clean.y).^2 .- (goal_width / 2)^2))
	clean.angle = abs.(clean.angle)  # symmetric, take absolute

	global shots_clean = clean

	p1 = scatter(clean.distance, clean.goal, alpha=0.5, legend=false, color=:steelblue,
		title="Distance au but vs But", xlabel="Distance au but (m)", ylabel="But (1=Oui)")
	p2 = scatter(clean.angle, clean.goal, alpha=0.5, legend=false, color=:coral,
		title="Angle au but vs But", xlabel="Angle au but (°)", ylabel="But (1=Oui)")

	plot(p1, p2, layout=(1, 2), size=(800, 350))
end

# ╔═╡ f8a9b0c1-0010-9d7e-5a6b-1c2d3e4f5a6b
md"""
**Observations :** les buts viennent de distances plus courtes et d'angles
plus larges (meilleure vue du but).
"""

# ╔═╡ f8a9b0c1-0011-9d7e-5a6b-1c2d3e4f5a6b
md"""
### Étape 3 — Préparer les données
"""

# ╔═╡ f8a9b0c1-0012-9d7e-5a6b-1c2d3e4f5a6b
begin
	n_xg = nrow(shots_clean)
	train_n = Int(round(0.8 * n_xg))
	train_ix = shuffle(1:n_xg)[1:train_n]
	test_ix = setdiff(1:n_xg, train_ix)

	df_train_xg = shots_clean[train_ix, :]
	df_test_xg  = shots_clean[test_ix, :]

	md"""
	- Entraînement : **$(nrow(df_train_xg)) tirs**
	- Test : **$(nrow(df_test_xg)) tirs**
	"""
end

# ╔═╡ f8a9b0c1-0013-9d7e-5a6b-1c2d3e4f5a6b
md"""
### Étape 4 — Entraîner le modèle logistique

Avec `GLM.jl` : `glm(formule, données, Binomial(), LogitLink())`.
"""

# ╔═╡ f8a9b0c1-0014-9d7e-5a6b-1c2d3e4f5a6b
let
	logit_model = glm(@formula(goal ~ distance + angle), df_train_xg,
		Binomial(), LogitLink())

	coeffs = coef(logit_model)

	md"""
	Modèle entraîné !

	**Coefficients :**
	- Intercept : **$(round(coeffs[1]; digits=4))**
	- Distance au but : **$(round(coeffs[2]; digits=4))** (négatif = plus loin → moins probable)
	- Angle au but : **$(round(coeffs[3]; digits=4))** (positif = plus large → plus probable)

	Ces signes correspondent à l'intuition footballistique !
	"""
end

# ╔═╡ f8a9b0c1-0015-9d7e-5a6b-1c2d3e4f5a6b
md"""
### Étape 5 — Faire des prédictions
"""

# ╔═╡ f8a9b0c1-0016-9d7e-5a6b-1c2d3e4f5a6b
let
	logit_model = glm(@formula(goal ~ distance + angle), df_train_xg,
		Binomial(), LogitLink())
	probs = predict(logit_model, df_test_xg)
	preds = Int.(probs .>= 0.5)

	DataFrame(
		distance = round.(df_test_xg.distance; digits=1),
		angle    = round.(df_test_xg.angle; digits=1),
		réel     = df_test_xg.goal,
		prob_xg  = round.(probs; digits=3),
		prédit   = preds,
	) |> (x -> first(x, 8))
end

# ╔═╡ f8a9b0c1-0017-9d7e-5a6b-1c2d3e4f5a6b
md"""
### Étape 6 — Prédire le xG de nouveaux tirs
"""

# ╔═╡ f8a9b0c1-0018-9d7e-5a6b-1c2d3e4f5a6b
let
	logit_model = glm(@formula(goal ~ distance + angle), df_train_xg,
		Binomial(), LogitLink())
	new_shots = DataFrame(
		distance = [5.0, 10.0, 20.0, 30.0],
		angle    = [30.0, 25.0, 15.0, 10.0],
		description = [
			"Proche, grand angle",
			"Dans la surface, bon angle",
			"Entrée de surface, angle étroit",
			"Longue distance, angle très étroit",
		])
	new_shots.xG = round.(predict(logit_model, new_shots); digits=3)
	select(new_shots, :description, :distance, :angle, :xG)
end

# ╔═╡ f8a9b0c1-0019-9d7e-5a6b-1c2d3e4f5a6b
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Appris comment la **régression logistique** utilise la fonction sigmoïde
2. Construit un modèle d'**Expected Goals (xG)**
3. Créé les variables **distance** et **angle** au but
4. Entraîné le modèle et **interprété les coefficients**
5. Fait des **prédictions** pour de nouveaux tirs

## Points clés

- La régression logistique prédit des **probabilités** (0 à 1)
- Les coefficients montrent l'**importance** et la **direction** de l'effet
- Les modèles xG quantifient la **qualité des tirs**
- Toujours **séparer** les données en entraînement/test

## Prochaine étape

Dans le prochain notebook, nous apprendrons à **évaluer** la performance
de notre modèle avec des métriques comme la précision, le rappel et l'AUC !
"""

# ╔═╡ f8a9b0c1-0020-9d7e-5a6b-1c2d3e4f5a6b
md"""
## Exercices

1. **Ajouter plus de variables** — essayer `shot.technique.name` ou
   `shot.body_part.name` comme variables supplémentaires.
2. **Match différent** — construire un modèle xG pour un autre match.
3. **Visualiser la frontière de décision** — tracer la frontière de
   décision du modèle dans l'espace 2D.
4. **Importance des variables** — quelle variable a le plus d'effet sur
   le xG ?
5. **Calcul manuel** — calculer le xG d'un tir à distance=15, angle=20
   en utilisant les coefficients.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
GLM = "1"
JSON3 = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─f8a9b0c1-0001-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0002-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0003-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0004-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0005-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0006-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0007-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0008-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0009-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0010-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0011-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0012-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0013-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0014-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0015-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0016-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0017-9d7e-5a6b-1c2d3e4f5a6b
# ╠═f8a9b0c1-0018-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0019-9d7e-5a6b-1c2d3e4f5a6b
# ╟─f8a9b0c1-0020-9d7e-5a6b-1c2d3e4f5a6b
# ╠═00000000-0000-0000-0000-000000000001
