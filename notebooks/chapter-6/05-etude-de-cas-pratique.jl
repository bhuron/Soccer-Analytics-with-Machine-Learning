### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ f2a3b4c5-0003-3d1e-9a0b-5c6d7e8f9a0b
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

# ╔═╡ f2a3b4c5-0001-3d1e-9a0b-5c6d7e8f9a0b
md"""
# Étude de cas pratique — Prédire les résultats de matchs

**Chapitre 6 · Techniques de régression pour l'analyse du football**

## Ce que vous allez apprendre

- Appliquer les techniques de régression à un problème complet
- Workflow de bout en bout : EDA → Feature engineering → Modélisation → Évaluation
- Construire et comparer plusieurs modèles de régression
- Faire des prédictions exploitables pour les résultats de matchs
- Interpréter les résultats dans un contexte footballistique
"""

# ╔═╡ f2a3b4c5-0002-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Imports et configuration
"""

# ╔═╡ f2a3b4c5-0004-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Le problème

**Question :** pouvons-nous prédire combien de buts une équipe va marquer
lors de son prochain match ?

**Pourquoi c'est important :**
- Planification tactique d'avant-match
- Analyse des marchés de paris
- Engagement des supporters (jeux de prédiction)
- Décisions de rotation d'effectif

**Approche :** utiliser des données historiques de performance d'équipe
pour construire un modèle prédictif.
"""

# ╔═╡ f2a3b4c5-0005-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 1 — Charger et explorer les données
"""

# ╔═╡ f2a3b4c5-0006-3d1e-9a0b-5c6d7e8f9a0b
begin
	teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]
	n_matches = 200

	match_df = DataFrame(
		team              = rand(teams, n_matches),
		shots_on_target   = rand(3:15, n_matches),
		possession        = round.(rand(35.0:0.1:70.0, n_matches); digits=1),
		pass_accuracy     = round.(rand(70.0:0.1:92.0, n_matches); digits=1),
		xg                = round.(rand(0.5:0.01:3.5, n_matches); digits=2),
		opponent_strength = round.(rand(0.3:0.01:0.9, n_matches); digits=2),
		home_advantage    = rand([0, 1], n_matches),
	)

	# Realistic goals
	match_df.goals = round.(Int, clamp.(
		match_df.xg .* 0.7 .+
		match_df.shots_on_target .* 0.08 .+
		match_df.home_advantage .* 0.3 .-
		match_df.opponent_strength .* 0.5 .+
		randn(n_matches) .* 0.4, 0, 6))

	first(match_df, 10)
end

# ╔═╡ f2a3b4c5-0007-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 2 — Analyse exploratoire (EDA)
"""

# ╔═╡ f2a3b4c5-0008-3d1e-9a0b-5c6d7e8f9a0b
let
	# Compute correlation matrix
	num_cols = [:shots_on_target, :possession, :pass_accuracy, :xg,
	            :opponent_strength, :home_advantage, :goals]
	corr_names = ["Tirs cadrés", "Possession", "Précision passes", "xG",
	              "Force adverse", "Domicile", "Buts"]
	n = length(num_cols)
	corr_mat = ones(n, n)
	for i in 1:n, j in 1:n
		corr_mat[i, j] = cor(match_df[!, num_cols[i]], match_df[!, num_cols[j]])
	end

	# 4-panel EDA
	p1 = heatmap(corr_mat, aspect_ratio=:equal, color=:coolwarm,
		xticks=(1:n, corr_names), yticks=(1:n, corr_names),
		xrotation=45, title="Matrice de corrélation", clims=(-1, 1))

	p2 = scatter(match_df.xg, match_df.goals, alpha=0.5, legend=false,
		color=:steelblue, title="Buts vs xG",
		xlabel="Buts attendus (xG)", ylabel="Buts réels")

	p3 = histogram(match_df.goals, bins=0:7, legend=false,
		color=:darkgreen, alpha=0.7, title="Distribution des buts",
		xlabel="Buts marqués", ylabel="Fréquence")

	home_avg = mean(match_df.goals[match_df.home_advantage .== 1])
	away_avg = mean(match_df.goals[match_df.home_advantage .== 0])
	p4 = bar(["Extérieur", "Domicile"], [away_avg, home_avg],
		legend=false, color=[:coral, :skyblue], alpha=0.7,
		title="Effet domicile", ylabel="Buts moyens")

	plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
end

# ╔═╡ f2a3b4c5-0009-3d1e-9a0b-5c6d7e8f9a0b
md"""
**Aperçus clés :**
- Le xG a la plus forte corrélation avec les buts
- L'avantage du terrain augmente les buts d'environ 0,3 en moyenne
- La force de l'adversaire est corrélée négativement avec les buts marqués
"""

# ╔═╡ f2a3b4c5-0010-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 3 — Feature engineering

Créer des variables supplémentaires pour améliorer les prédictions.
"""

# ╔═╡ f2a3b4c5-0011-3d1e-9a0b-5c6d7e8f9a0b
begin
	match_df.xg_shots       = match_df.xg .* match_df.shots_on_target
	match_df.home_vs_strong = match_df.home_advantage .* (1 .- match_df.opponent_strength)
	match_df.shot_efficiency = match_df.xg ./ (match_df.shots_on_target .+ 1)

	# Team strength encoding
	team_avg = combine(groupby(match_df, :team), :goals => mean => :team_strength)
	match_df = leftjoin(match_df, team_avg, on=:team)

	match_df
end

# ╔═╡ f2a3b4c5-0012-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 4 — Préparer les données
"""

# ╔═╡ f2a3b4c5-0013-3d1e-9a0b-5c6d7e8f9a0b
begin
	feature_cols = [
		:shots_on_target, :possession, :pass_accuracy, :xg,
		:opponent_strength, :home_advantage, :team_strength,
		:xg_shots, :home_vs_strong, :shot_efficiency]

	X_all_cs = Matrix(match_df[!, feature_cols])
	y_all_cs = Vector(match_df[!, :goals])

	n_train_cs = Int(round(0.75 * n_matches))
	train_idx_cs = shuffle(1:n_matches)[1:n_train_cs]
	test_idx_cs  = setdiff(1:n_matches, train_idx_cs)

	X_tr = X_all_cs[train_idx_cs, :]
	y_tr = y_all_cs[train_idx_cs]
	X_te = X_all_cs[test_idx_cs, :]
	y_te = y_all_cs[test_idx_cs]

	md"""
	- Entraînement : **$(n_train_cs) matchs** (75 %)
	- Test : **$(n_matches - n_train_cs) matchs** (25 %)
	- **$(length(feature_cols)) variables**
	"""
end

# ╔═╡ f2a3b4c5-0014-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 5 — Construire et comparer les modèles

Trois approches : régression linéaire, Poisson et KNN.
"""

# ╔═╡ f2a3b4c5-0015-3d1e-9a0b-5c6d7e8f9a0b
function knn_predict_cs(X_train, y_train, X_query, k)
	dists = [sqrt(sum((X_query[i, :] .- X_train[j, :]).^2)) for i in 1:size(X_query,1), j in 1:size(X_train,1)]
	preds = zeros(size(X_query, 1))
	for i in 1:size(X_query, 1)
		neighbors = sortperm(dists[i, :])[1:k]
		preds[i] = mean(y_train[neighbors])
	end
	return preds
end

# ╔═╡ f2a3b4c5-0016-3d1e-9a0b-5c6d7e8f9a0b
let
	df_tr = DataFrame(X_tr, feature_cols)
	df_tr.goals = y_tr
	df_te = DataFrame(X_te, feature_cols)

	# 1. Linear regression
	lin_model = lm(@formula(goals ~ shots_on_target + possession + pass_accuracy + xg +
		opponent_strength + home_advantage + team_strength +
		xg_shots + home_vs_strong + shot_efficiency), df_tr)
	lin_pred = predict(lin_model, df_te)

	# 2. Poisson regression
	pois_model = glm(@formula(goals ~ shots_on_target + possession + pass_accuracy + xg +
		opponent_strength + home_advantage + team_strength +
		xg_shots + home_vs_strong + shot_efficiency), df_tr, Poisson(), LogLink())
	pois_pred = predict(pois_model, df_te)

	# 3. KNN (K=7)
	X_mean = mean(X_tr; dims=1)
	X_std  = std(X_tr; dims=1)
	X_tr_s = (X_tr .- X_mean) ./ X_std
	X_te_s = (X_te .- X_mean) ./ X_std
	knn_pred = knn_predict_cs(X_tr_s, y_tr, X_te_s, 7)

	global lin_pred, pois_pred, knn_pred

	# 3-panel prediction vs actual
	p1 = scatter(y_te, lin_pred, alpha=0.6, legend=false, color=:steelblue,
		title="Régression linéaire", xlabel="Buts réels", ylabel="Buts prédits",
		xlims=(-0.5, 6.5), ylims=(-0.5, 6.5))
	plot!(p1, [0, 6], [0, 6], color=:red, linestyle=:dash, linewidth=2)

	p2 = scatter(y_te, pois_pred, alpha=0.6, legend=false, color=:coral,
		title="Régression de Poisson", xlabel="Buts réels", ylabel="Buts prédits",
		xlims=(-0.5, 6.5), ylims=(-0.5, 6.5))
	plot!(p2, [0, 6], [0, 6], color=:red, linestyle=:dash, linewidth=2)

	p3 = scatter(y_te, knn_pred, alpha=0.6, legend=false, color=:darkgreen,
		title="KNN (K=7)", xlabel="Buts réels", ylabel="Buts prédits",
		xlims=(-0.5, 6.5), ylims=(-0.5, 6.5))
	plot!(p3, [0, 6], [0, 6], color=:red, linestyle=:dash, linewidth=2)

	plot(p1, p2, p3, layout=(1, 3), size=(950, 300))
end

# ╔═╡ f2a3b4c5-0017-3d1e-9a0b-5c6d7e8f9a0b
md"""
Les points proches de la ligne rouge = meilleures prédictions.
"""

# ╔═╡ f2a3b4c5-0018-3d1e-9a0b-5c6d7e8f9a0b
let
	function metrics(y_true, y_pred, name)
		r2 = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)
		rmse = sqrt(mean((y_true .- y_pred).^2))
		mae = mean(abs.(y_true .- y_pred))
		(name, r2, rmse, mae)
	end

	results = DataFrame(
		modèle = String[], R² = Float64[], RMSE = Float64[], MAE = Float64[])
	for (name, preds) in [("Linéaire", lin_pred), ("Poisson", pois_pred), ("KNN", knn_pred)]
		push!(results, metrics(y_te, preds, name))
	end
	results
end

# ╔═╡ f2a3b4c5-0019-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 6 — Interpréter le meilleur modèle

Examinons les coefficients du modèle linéaire pour leur interprétabilité.
"""

# ╔═╡ f2a3b4c5-0020-3d1e-9a0b-5c6d7e8f9a0b
let
	df_tr = DataFrame(X_tr, feature_cols)
	df_tr.goals = y_tr
	lin_model = lm(@formula(goals ~ shots_on_target + possession + pass_accuracy + xg +
		opponent_strength + home_advantage + team_strength +
		xg_shots + home_vs_strong + shot_efficiency), df_tr)

	coeffs = coef(lin_model)
	c_names = ["(intercept)", [string(f) for f in feature_cols]...]

	# Sort by absolute value
	idx = sortperm(abs.(coeffs[2:end]), rev=true)
	sorted_names = [c_names[1]; [string(feature_cols[i]) for i in idx]]
	sorted_coeffs = [coeffs[1]; [coeffs[i+1] for i in idx]]

	n = length(sorted_names)
	bar(1:n, sorted_coeffs,
		legend=false, color=[c > 0 ? :green : :coral for c in sorted_coeffs],
		alpha=0.7, title="Impact des variables sur les buts marqués",
		ylabel="Valeur du coefficient", xticks=(1:n, sorted_names), xrotation=45)
	hline!([0], color=:black, linewidth=0.8)
end

# ╔═╡ f2a3b4c5-0021-3d1e-9a0b-5c6d7e8f9a0b
md"""
**Interprétation :**
- Barres vertes : impact positif sur les buts
- Barres rouges : impact négatif
- Barres plus grandes : influence plus forte
- Le xG et l'avantage du terrain sont les facteurs dominants
"""

# ╔═╡ f2a3b4c5-0022-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Étape 7 — Prédictions pour les prochains matchs
"""

# ╔═╡ f2a3b4c5-0023-3d1e-9a0b-5c6d7e8f9a0b
let
	upcoming = DataFrame(
		shots_on_target   = [10, 8, 12],
		possession        = [55.0, 48.0, 62.0],
		pass_accuracy     = [85.0, 78.0, 88.0],
		xg                = [2.1, 1.5, 2.8],
		opponent_strength = [0.6, 0.8, 0.4],
		home_advantage    = [1, 0, 1],
		team_strength     = [2.2, 1.8, 2.5])
	upcoming.xg_shots       = upcoming.xg .* upcoming.shots_on_target
	upcoming.home_vs_strong = upcoming.home_advantage .* (1 .- upcoming.opponent_strength)
	upcoming.shot_efficiency = upcoming.xg ./ (upcoming.shots_on_target .+ 1)

	# Re-train models on full data
	df_all = DataFrame(X_all_cs, feature_cols)
	df_all.goals = y_all_cs

	lin_final = lm(@formula(goals ~ shots_on_target + possession + pass_accuracy + xg +
		opponent_strength + home_advantage + team_strength +
		xg_shots + home_vs_strong + shot_efficiency), df_all)
	pois_final = glm(@formula(goals ~ shots_on_target + possession + pass_accuracy + xg +
		opponent_strength + home_advantage + team_strength +
		xg_shots + home_vs_strong + shot_efficiency), df_all, Poisson(), LogLink())

	X_mean_f = mean(X_all_cs; dims=1)
	X_std_f  = std(X_all_cs; dims=1)
	X_up_s   = (Matrix(upcoming[!, feature_cols]) .- X_mean_f) ./ X_std_f
	knn_up   = knn_predict_cs((X_all_cs .- X_mean_f) ./ X_std_f, y_all_cs, X_up_s, 7)

	DataFrame(
		match      = ["Match 1 (domicile)", "Match 2 (extérieur)", "Match 3 (domicile)"],
		linéaire   = round.(predict(lin_final, upcoming); digits=2),
		poisson    = round.(predict(pois_final, upcoming); digits=2),
		knn        = round.(knn_up; digits=2),
	)
end

# ╔═╡ f2a3b4c5-0024-3d1e-9a0b-5c6d7e8f9a0b
md"""
Ces prédictions peuvent informer :
- Les décisions tactiques (configuration défensive vs offensive)
- La rotation de l'effectif (reposer les joueurs clés dans les matchs à faible score)
- L'engagement des supporters (concours de prédiction)
"""

# ╔═╡ f2a3b4c5-0025-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Considérations pratiques

**Ce que nous avons appris :**

1. **Sélection de modèle :**
   - Linéaire : simple, interprétable, bonne référence
   - Poisson : théoriquement adapté aux données de comptage
   - KNN : flexible mais moins interprétable

2. **Le feature engineering compte :**
   - Les termes d'interaction capturent les relations complexes
   - La connaissance du domaine améliore les variables
   - L'encodage de la force d'équipe ajoute du contexte

3. **Limitations :**
   - Ne peut pas prédire les événements rares (cartons rouges, blessures)
   - Les données historiques peuvent ne pas refléter la forme actuelle
   - Facteurs externes (météo, motivation) non capturés

4. **Déploiement réel :**
   - Mettre à jour le modèle régulièrement avec de nouvelles données
   - Surveiller la précision des prédictions dans le temps
   - Combiner avec le jugement d'expert
   - Utiliser des intervalles de prédiction, pas seulement des estimations ponctuelles
"""

# ╔═╡ f2a3b4c5-0026-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Récapitulatif

Dans cette étude de cas, nous avons :

1. Défini un problème clair
2. Effectué une EDA complète (matrice de corrélation, distributions)
3. Créé des variables significatives (interactions, force d'équipe)
4. Construit et comparé trois modèles (linéaire, Poisson, KNN)
5. Évalué rigoureusement les performances
6. Interprété les résultats en contexte footballistique
7. Fait des prédictions exploitables
8. Discuté des considérations pratiques

## Points clés

- Le **workflow de bout en bout** est crucial pour les vrais projets
- Le **feature engineering** compte souvent plus que le choix du modèle
- La **comparaison des modèles** doit être systématique
- L'**interprétation** rend les prédictions exploitables
- Les **limites pratiques** doivent être reconnues
- La **connaissance du domaine** améliore chaque étape

## Félicitations !

Vous avez terminé les notebooks du chapitre 6 ! Vous avez maintenant une
base solide en techniques de régression pour l'analyse du football.
"""

# ╔═╡ f2a3b4c5-0027-3d1e-9a0b-5c6d7e8f9a0b
md"""
## Exercices

1. **Plus de variables** — inclure des métriques défensives, la forme récente,
   l'historique des confrontations.
2. **Split temporel** — utiliser une validation croisée temporelle au lieu
   d'un split aléatoire.
3. **Modèle d'ensemble** — combiner les prédictions de plusieurs modèles.
4. **Intervalles de prédiction** — ajouter des intervalles de confiance.
5. **Données réelles** — appliquer à des données réelles de StatsBomb.
6. **Deux équipes** — prédire les buts des deux équipes et déterminer le
   résultat du match.
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
# ╟─f2a3b4c5-0001-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0002-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0003-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0004-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0005-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0006-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0007-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0008-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0009-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0010-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0011-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0012-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0013-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0014-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0015-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0016-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0017-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0018-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0019-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0020-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0021-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0022-3d1e-9a0b-5c6d7e8f9a0b
# ╠═f2a3b4c5-0023-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0024-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0025-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0026-3d1e-9a0b-5c6d7e8f9a0b
# ╟─f2a3b4c5-0027-3d1e-9a0b-5c6d7e8f9a0b
# ╠═00000000-0000-0000-0000-000000000001
