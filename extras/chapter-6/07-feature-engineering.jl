### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ b4c5d6e7-0003-5f3a-1c2d-7e8f9a0b1c2d
begin
	using DataFrames
	using Statistics
	using Random
	using LinearAlgebra
	using GLM
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ b4c5d6e7-0001-5f3a-1c2d-7e8f9a0b1c2d
md"""
# Feature Engineering pour de meilleures prédictions

**Chapitre 6 · Techniques de régression pour l'analyse du football — EXTRA**

## Ce que vous allez apprendre

- Créer des variables d'interaction pour capturer les effets combinés
- Générer des variables polynomiales pour les relations non linéaires
- Encoder efficacement les variables catégorielles
- Construire des variables décalées à partir de données temporelles
- Créer des variables spécifiques au domaine du football
- Sélectionner les variables pertinentes
- Éviter les pièges courants du feature engineering
"""

# ╔═╡ b4c5d6e7-0002-5f3a-1c2d-7e8f9a0b1c2d
md"""
## Imports et configuration
"""

# ╔═╡ b4c5d6e7-0004-5f3a-1c2d-7e8f9a0b1c2d
md"""
## Pourquoi le feature engineering est crucial

*« Trouver les bonnes variables est difficile, prend du temps et nécessite
des connaissances expertes. Le machine learning appliqué, c'est
essentiellement du feature engineering. »* — Andrew Ng

**Dans l'analyse du football :**
- Les statistiques brutes (tirs, passes) ne sont qu'un point de départ
- Les **effets d'interaction** comptent (ex : tirs × précision)
- Le **contexte** est crucial (domicile vs extérieur, force adverse)
- Les **motifs temporels** révèlent la forme et l'élan
- La **connaissance du domaine** crée les meilleures variables
"""

# ╔═╡ b4c5d6e7-0005-5f3a-1c2d-7e8f9a0b1c2d
begin
	teams_fe = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United",
	            "Tottenham", "Leicester", "West Ham", "Everton", "Wolves"]
	n_fe = 300

	df_fe = DataFrame(
		match_id        = 1:n_fe,
		team            = rand(teams_fe, n_fe),
		opponent        = rand(teams_fe, n_fe),
		home            = rand([0, 1], n_fe),
		shots           = rand(5:20, n_fe),
		shots_on_target = rand(2:12, n_fe),
		possession      = round.(rand(35.0:0.1:70.0, n_fe); digits=1),
		pass_accuracy   = round.(rand(70.0:0.1:92.0, n_fe); digits=1),
		xg              = round.(rand(0.5:0.01:3.5, n_fe); digits=2),
		opponent_xg     = round.(rand(0.5:0.01:3.5, n_fe); digits=2),
		match_week      = rand(1:38, n_fe),
	)

	df_fe.goals = round.(Int, clamp.(df_fe.xg .* 0.8 .+ df_fe.shots_on_target .* 0.05 .+
		df_fe.home .* 0.3 .+ randn(n_fe) .* 0.5, 0, 6))

	first(df_fe, 5)
end

# ╔═╡ b4c5d6e7-0006-5f3a-1c2d-7e8f9a0b1c2d
md"""
## 1. Variables d'interaction

Capturer les effets combinés de deux ou plusieurs variables.
"""

# ╔═╡ b4c5d6e7-0007-5f3a-1c2d-7e8f9a0b1c2d
let
	df_fe.shot_quality        = df_fe.shots_on_target ./ (df_fe.shots .+ 1)
	df_fe.effective_possession = df_fe.possession .* df_fe.pass_accuracy ./ 100
	df_fe.xg_difference       = df_fe.xg .- df_fe.opponent_xg
	df_fe.shot_efficiency     = df_fe.xg ./ (df_fe.shots .+ 1)
	df_fe.attacking_threat    = df_fe.shots_on_target .* df_fe.xg

	p1 = scatter(df_fe.shots_on_target, df_fe.goals, alpha=0.3, legend=false, color=:steelblue,
		title="Variable brute", xlabel="Tirs cadrés", ylabel="Buts")
	p2 = scatter(df_fe.attacking_threat, df_fe.goals, alpha=0.3, legend=false, color=:darkgreen,
		title="Variable d'interaction", xlabel="Menace offensive (Tirs cadrés × xG)", ylabel="Buts")

	plot(p1, p2, layout=(1, 2), size=(800, 350))
end

# ╔═╡ b4c5d6e7-0008-5f3a-1c2d-7e8f9a0b1c2d
md"""
Les variables d'interaction montrent souvent des corrélations plus fortes
avec la cible que les variables brutes seules.
"""

# ╔═╡ b4c5d6e7-0009-5f3a-1c2d-7e8f9a0b1c2d
md"""
## 2. Variables polynomiales

Capturer les relations non linéaires en ajoutant des termes au carré, cube…
"""

# ╔═╡ b4c5d6e7-0010-5f3a-1c2d-7e8f9a0b1c2d
let
	X_poly = DataFrame(
		xg           = df_fe.xg,
		xg_squared   = df_fe.xg.^2,
		sot          = df_fe.shots_on_target,
		sot_squared  = df_fe.shots_on_target.^2,
		xg_times_sot = df_fe.xg .* df_fe.shots_on_target,
	)

	X_poly.goals = df_fe.goals

	model_orig = lm(@formula(goals ~ xg + sot), X_poly)
	model_poly = lm(@formula(goals ~ xg + xg_squared + sot + sot_squared + xg_times_sot), X_poly)

	f_orig = X_poly[!, [:xg, :sot]]
	f_poly = X_poly[!, [:xg, :xg_squared, :sot, :sot_squared, :xg_times_sot]]

	md"""
	**Comparaison des modèles :**
	- R² avec variables originales : **$(round(r²(model_orig); digits=3))**
	- R² avec variables polynomiales : **$(round(r²(model_poly); digits=3))**
	- Les termes au carré capturent la non-linéarité de la relation buts/xG.
	"""
end

# ╔═╡ b4c5d6e7-0011-5f3a-1c2d-7e8f9a0b1c2d
md"""
## 3. Encodage des variables catégorielles

Les modèles ont besoin de nombres, pas de texte. Trois méthodes :
- **Target encoding** — remplacer par la moyenne de la cible
- **One-hot encoding** — colonnes binaires pour chaque catégorie
- **Frequency encoding** — remplacer par la fréquence d'apparition
"""

# ╔═╡ b4c5d6e7-0012-5f3a-1c2d-7e8f9a0b1c2d
let
	# Target encoding: average goals per team
	team_avg = combine(groupby(df_fe, :team), :goals => mean => :team_strength)
	opp_avg  = combine(groupby(df_fe, :opponent), :goals => mean => :opponent_strength_fe)
	rename!(opp_avg, :opponent => :team)
	df_fe2 = leftjoin(df_fe, team_avg, on=:team)
	df_fe2 = leftjoin(df_fe2, rename(opp_avg, :team => :opponent), on=:opponent)
	rename!(df_fe2, :opponent_strength_fe => :opponent_def_strength)

	# Show top teams by strength
	sort(combine(groupby(df_fe2, :team), :team_strength => first => :strength), :strength, rev=true)
end

# ╔═╡ b4c5d6e7-0013-5f3a-1c2d-7e8f9a0b1c2d
md"""
## 4. Variables décalées (temporelles)

Utiliser les performances passées pour prédire les résultats futurs.
"""

# ╔═╡ b4c5d6e7-0014-5f3a-1c2d-7e8f9a0b1c2d
let
	df_sorted = sort(df_fe, [:team, :match_week])

	# Manual lagged features per team
	df_sorted.goals_last_1 = Vector{Union{Missing,Int}}(missing, nrow(df_sorted))
	df_sorted.goals_last_3_avg = Vector{Union{Missing,Float64}}(missing, nrow(df_sorted))

	for team in unique(df_sorted.team)
		idx = findall(df_sorted.team .== team)
		g = df_sorted.goals[idx]
		for i in 1:length(g)
			if i > 1
				df_sorted.goals_last_1[idx[i]] = g[i-1]
			end
			if i > 3
				df_sorted.goals_last_3_avg[idx[i]] = mean(g[i-3:i-1])
			end
		end
	end

	# Form indicator
	df_sorted.goals_last_5_avg = Vector{Union{Missing,Float64}}(missing, nrow(df_sorted))
	for team in unique(df_sorted.team)
		idx = findall(df_sorted.team .== team)
		g = df_sorted.goals[idx]
		for i in 1:length(g)
			if i > 5
				df_sorted.goals_last_5_avg[idx[i]] = mean(g[i-5:i-1])
			end
		end
	end
	df_sorted.recent_form = df_sorted.goals_last_3_avg .- df_sorted.goals_last_5_avg

	# Show Arsenal as example
	ars = subset(df_sorted, :team => ByRow(==("Arsenal")); skipmissing=true)
	select(first(ars, 8), :match_week, :goals, :goals_last_1, :goals_last_3_avg, :recent_form)
end

# ╔═╡ b4c5d6e7-0015-5f3a-1c2d-7e8f9a0b1c2d
md"""
## 5. Variables spécifiques au football

Basées sur la connaissance tactique du domaine.
"""

# ╔═╡ b4c5d6e7-0016-5f3a-1c2d-7e8f9a0b1c2d
let
	df_fe.defensive_solidity   = 1 ./ (df_fe.opponent_xg .+ 0.1)
	df_fe.attacking_efficiency = df_fe.xg ./ (df_fe.shots .+ 1)
	df_fe.possession_quality   = (df_fe.possession .* df_fe.pass_accuracy) ./ 100
	df_fe.match_importance     = df_fe.match_week ./ 38
	df_fe.expected_goal_diff   = df_fe.xg .- df_fe.opponent_xg

	# Compute correlations with goals
	domain_cols = [:defensive_solidity, :attacking_efficiency, :possession_quality,
	              :match_importance, :expected_goal_diff]
	corrs = [cor(df_fe[!, c], df_fe.goals) for c in domain_cols]
	DataFrame(variable=string.(domain_cols), corrélation_avec_buts=round.(corrs; digits=3))
end

# ╔═╡ b4c5d6e7-0017-5f3a-1c2d-7e8f9a0b1c2d
md"""
## 6. Sélection de variables

Trop de variables peut causer du surapprentissage. Trois méthodes :
1. **Filtrage** — sélection basée sur des tests statistiques
2. **Wrapper** — sélection basée sur la performance du modèle
3. **Embarquée** — sélection pendant l'entraînement
"""

# ╔═╡ b4c5d6e7-0018-5f3a-1c2d-7e8f9a0b1c2d
let
	candidates = [:shots, :shots_on_target, :possession, :pass_accuracy, :xg, :opponent_xg,
		:home, :shot_quality, :effective_possession, :xg_difference, :shot_efficiency,
		:attacking_threat, :defensive_solidity, :attacking_efficiency, :expected_goal_diff]

	# Filter: select top 8 by absolute correlation with goals
	corrs = [abs(cor(df_fe[!, c], df_fe.goals)) for c in candidates]
	idx = sortperm(corrs, rev=true)[1:8]
	best = [string(candidates[i]) for i in idx]

	md"""
	**Top 8 variables par corrélation absolue avec les buts :**
	$(join(["$(i). $(best[i])" for i in 1:8], "  \n"))
	"""
end

# ╔═╡ b4c5d6e7-0019-5f3a-1c2d-7e8f9a0b1c2d
md"""
## Pièges à éviter

### 1. Fuite de données (data leakage)
Utiliser des informations qui ne seraient pas disponibles au moment de
la prédiction. **❌ Ne jamais utiliser `.shift(-1)`** — ce sont des
données futures !

### 2. Multicolinéarité
Des variables fortement corrélées entre elles rendent les modèles
instables. Vérifier les paires avec |r| > 0,8.

### 3. Surapprentissage
Trop de variables → le modèle mémorise au lieu d'apprendre.
Utiliser la validation croisée pour détecter.
"""

# ╔═╡ b4c5d6e7-0020-5f3a-1c2d-7e8f9a0b1c2d
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Créé des variables d'interaction pour les effets combinés
2. Généré des variables polynomiales pour la non-linéarité
3. Encodé des variables catégorielles (target, one-hot, fréquence)
4. Construit des variables décalées à partir de données temporelles
5. Conçu des variables spécifiques au football
6. Appliqué la sélection de variables par corrélation

## Points clés

- **Feature engineering > Choix du modèle** dans la plupart des cas
- **Variables d'interaction** — capturent les effets combinés
- **Variables polynomiales** — gèrent les relations non linéaires
- **Variables décalées** — capturent les motifs temporels
- **Connaissance du domaine** — crée les variables les plus puissantes
- **Sélection de variables** — prévient le surapprentissage
- **Éviter la fuite de données** à tout prix

## Exercices

1. **Interactions triples** — essayer des interactions à 3 voies
2. **Statistiques glissantes** — créer min/max glissants, pas seulement des moyennes
3. **Variables adverses** — historique des confrontations, matchups tactiques
4. **Variables temporelles** — jour de la semaine, mois, période de la saison
5. **Importance des variables** — utiliser une forêt aléatoire pour classer les variables
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
GLM = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─b4c5d6e7-0001-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0002-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0003-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0004-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0005-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0006-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0007-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0008-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0009-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0010-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0011-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0012-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0013-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0014-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0015-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0016-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0017-5f3a-1c2d-7e8f9a0b1c2d
# ╠═b4c5d6e7-0018-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0019-5f3a-1c2d-7e8f9a0b1c2d
# ╟─b4c5d6e7-0020-5f3a-1c2d-7e8f9a0b1c2d
# ╠═00000000-0000-0000-0000-000000000001
