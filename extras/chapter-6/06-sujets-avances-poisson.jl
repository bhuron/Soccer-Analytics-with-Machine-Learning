### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ a3b4c5d6-0003-4e2f-0b1c-6d7e8f9a0b1c
begin
	using DataFrames
	using Statistics
	using Random
	using Distributions
	using GLM
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ a3b4c5d6-0001-4e2f-0b1c-6d7e8f9a0b1c
md"""
# Sujets avancés — Régression de Poisson

**Chapitre 6 · Techniques de régression pour l'analyse du football — EXTRA**

## Ce que vous allez apprendre

- Comprendre la surdispersion dans les données de comptage
- Implémenter la régression binomiale négative
- Gérer l'inflation de zéros dans les données de football
- Construire des modèles hurdle en deux étapes
- Diagnostiquer les violations du modèle de Poisson
"""

# ╔═╡ a3b4c5d6-0002-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Imports et configuration
"""

# ╔═╡ a3b4c5d6-0004-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Le problème — quand Poisson ne suffit pas

**Hypothèse standard de Poisson :** Variance = Moyenne

**Réalité dans le football :**
- Certaines équipes marquent régulièrement beaucoup/peu (variance supplémentaire)
- Beaucoup de matchs se terminent 0-0 (inflation de zéros)
- Les buts arrivent par grappes (surdispersion)

**Solutions :**
1. **Binomiale négative** — gère la surdispersion
2. **Poisson avec inflation de zéros (ZIP)** — gère l'excès de zéros
3. **Modèles hurdle** — processus en deux étapes
"""

# ╔═╡ a3b4c5d6-0005-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Surdispersion — quand la variance dépasse la moyenne

La **surdispersion** se produit quand la variance des données est supérieure
à ce que le modèle de Poisson prévoit.
"""

# ╔═╡ a3b4c5d6-0006-4e2f-0b1c-6d7e8f9a0b1c
begin
	n_od = 200
	shots_od  = rand(5:15, n_od)
	team_eff  = rand([-0.5, 0.0, 0.5], n_od)
	goals_od  = rand.(Poisson.(shots_od .* 0.15 .+ team_eff .+ 1))

	df_od = DataFrame(shots=shots_od, goals=goals_od)

	mean_g = mean(df_od.goals)
	var_g  = var(df_od.goals)
	ratio  = var_g / mean_g

	md"""
	- Moyenne des buts : **$(round(mean_g; digits=3))**
	- Variance : **$(round(var_g; digits=3))**
	- Ratio variance/moyenne : **$(round(ratio; digits=3))**

	$(ratio > 1.2 ? "⚠️ Surdispersion détectée (ratio > 1,2) !" : ratio > 0.8 ? "✅ Poisson approprié" : "Sous-dispersion (rare)")
	"""
end

# ╔═╡ a3b4c5d6-0007-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Comparer Poisson vs Binomiale négative
"""

# ╔═╡ a3b4c5d6-0008-4e2f-0b1c-6d7e8f9a0b1c
let
	pois_model = glm(@formula(goals ~ shots), df_od, Poisson(), LogLink())
	nb_model   = glm(@formula(goals ~ shots), df_od, NegativeBinomial(), LogLink())

	df_od.pois_pred = predict(pois_model, df_od)
	df_od.nb_pred   = predict(nb_model, df_od)

	global pois_aic = aic(pois_model)
	global nb_aic   = aic(nb_model)

	p1 = scatter(df_od.goals, df_od.pois_pred, alpha=0.5, legend=false, color=:steelblue,
		title="Poisson (AIC=$(round(pois_aic; digits=1)))",
		xlabel="Buts réels", ylabel="Buts prédits")
	plot!(p1, [0, 8], [0, 8], color=:red, linestyle=:dash, linewidth=2)

	p2 = scatter(df_od.goals, df_od.nb_pred, alpha=0.5, legend=false, color=:darkgreen,
		title="Bin. négative (AIC=$(round(nb_aic; digits=1)))",
		xlabel="Buts réels", ylabel="Buts prédits")
	plot!(p2, [0, 8], [0, 8], color=:red, linestyle=:dash, linewidth=2)

	plot(p1, p2, layout=(1, 2), size=(800, 350))
end

# ╔═╡ a3b4c5d6-0009-4e2f-0b1c-6d7e8f9a0b1c
md"""
$(nb_aic < pois_aic ? "✅ La binomiale négative est meilleure (AIC plus bas) — surdispersion confirmée !" : "✅ Poisson suffit — pas de surdispersion significative.")

La binomiale négative fournit des intervalles de prédiction plus larges,
tenant compte de la variabilité supplémentaire dans les données.
"""

# ╔═╡ a3b4c5d6-0010-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Inflation de zéros — trop de 0-0

**Problème :** certains matchs ont plus de 0-0 que ce qu'un modèle de
Poisson prédirait.

**Causes dans le football :** tactiques défensives, mauvais temps,
matchs à faible enjeu, équipes déséquilibrées qui jouent prudemment.
"""

# ╔═╡ a3b4c5d6-0011-4e2f-0b1c-6d7e8f9a0b1c
begin
	n_zi = 200
	shots_zi    = rand(5:15, n_zi)
	defensive   = rand(Binomial(1, 0.3), n_zi)  # 30% défensifs
	goals_zi    = [d == 1 ? 0 : rand(Poisson(s * 0.15)) for (s, d) in zip(shots_zi, defensive)]
	df_zi = DataFrame(shots=shots_zi, goals=goals_zi)

	obs_zeros  = count(==(0), df_zi.goals)
	mean_zi    = mean(df_zi.goals)
	exp_zeros  = n_zi * pdf(Poisson(mean_zi), 0)

	md"""
	- Zéros observés : **$obs_zeros ($(round(100 * obs_zeros / n_zi; digits=1)) %)**
	- Zéros attendus (Poisson) : **$(round(Int, exp_zeros)) ($(round(100 * exp_zeros / n_zi; digits=1)) %)**
	- $(obs_zeros > 1.2 * exp_zeros ? "⚠️ Inflation de zéros détectée !" : "✅ Pas d'inflation de zéros significative.")
	"""
end

# ╔═╡ a3b4c5d6-0012-4e2f-0b1c-6d7e8f9a0b1c
let
	max_g = maximum(df_zi.goals)
	obs_counts = [count(==(k), df_zi.goals) for k in 0:max_g]
	exp_counts = [n_zi * pdf(Poisson(mean_zi), k) for k in 0:max_g]

	bar(0:max_g .- 0.2, obs_counts, bar_width=0.4, label="Observé",
		color=:steelblue, alpha=0.7, title="Distribution observée vs attendue (Poisson)",
		xlabel="Nombre de buts", ylabel="Fréquence")
	bar!(0:max_g .+ 0.2, exp_counts, bar_width=0.4, label="Attendu (Poisson)",
		color=:coral, alpha=0.7)
end

# ╔═╡ a3b4c5d6-0013-4e2f-0b1c-6d7e8f9a0b1c
md"""
Remarquez l'excès de zéros dans les données observées (barre bleue à 0).
"""

# ╔═╡ a3b4c5d6-0014-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Modèles hurdle — approche en deux étapes

**Idée :** modéliser le processus en deux étapes :
1. **Étape 1 :** y aura-t-il des buts ? (Binaire : Oui/Non)
2. **Étape 2 :** si oui, combien ? (Modèle de comptage pour valeurs > 0)

**Quand l'utiliser :** quand le processus de marquer le premier but est
fondamentalement différent de marquer des buts supplémentaires.
"""

# ╔═╡ a3b4c5d6-0015-4e2f-0b1c-6d7e8f9a0b1c
let
	df_zi.any_goals = Int.(df_zi.goals .> 0)

	# Stage 1: logistic regression
	logit_model = glm(@formula(any_goals ~ shots), df_zi, Binomial(), LogitLink())
	prob_any = predict(logit_model, df_zi)

	# Stage 2: Poisson on positive goals only
	df_pos = subset(df_zi, :goals => ByRow(>(0)); skipmissing=true)
	pois_pos = glm(@formula(goals ~ shots), df_pos, Poisson(), LogLink())

	# Combined prediction
	pos_pred = predict(pois_pos, df_zi)
	hurdle_pred = prob_any .* pos_pred

	md"""
	**Étape 1 — Régression logistique :**
	- Probabilité moyenne de marquer : **$(round(mean(prob_any); digits=3))**

	**Étape 2 — Poisson sur buts positifs :**
	- Coefficient tirs : **$(round(coef(pois_pos)[2]; digits=4))**

	**Prédiction combinée :**
	- `Prédiction = P(marquer) × E(buts | marquer)`
	"""
end

# ╔═╡ a3b4c5d6-0016-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Test d'adéquation (goodness-of-fit)

Comment savoir si notre modèle de Poisson s'ajuste bien ?
"""

# ╔═╡ a3b4c5d6-0017-4e2f-0b1c-6d7e8f9a0b1c
let
	pois_simple = glm(@formula(goals ~ shots), df_od, Poisson(), LogLink())
	resids = residuals(pois_simple, type=:pearson)
	chi2 = sum(resids.^2)
	df_resid = nrow(df_od) - length(coef(pois_simple))
	r = chi2 / df_resid

	md"""
	**Test d'adéquation :**
	- χ² de Pearson : **$(round(chi2; digits=1))**
	- Degrés de liberté : **$df_resid**
	- Ratio χ²/df : **$(round(r; digits=2))**

	**Interprétation :**
	$(r > 1.5 ? "⚠️ Surdispersion — envisager la binomiale négative" :
	  r > 0.7 ? "✅ Bon ajustement du modèle de Poisson" :
	  "⚠️ Sous-dispersion — vérifier la spécification du modèle")
	"""
end

# ╔═╡ a3b4c5d6-0018-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Guide de sélection du modèle

| Situation | Modèle recommandé |
|---|---|
| **Variance ≈ Moyenne** | Poisson standard |
| **Variance > Moyenne** | Binomiale négative |
| **Excès de zéros** | Poisson avec inflation de zéros (ZIP) |
| **Processus en deux étapes** | Modèle hurdle |
| **Surdispersion + zéros** | Binomiale négative avec inflation de zéros |

**Arbre de décision :**
1. Vérifier le ratio variance/moyenne → si > 1,5, binomiale négative
2. Vérifier la proportion de zéros → si excès, ZIP ou hurdle
3. Comparer les modèles avec l'AIC → le plus bas est le meilleur
4. Valider sur données de test → généralisation
"""

# ╔═╡ a3b4c5d6-0019-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Exemple pratique — choisir le bon modèle
"""

# ╔═╡ a3b4c5d6-0020-4e2f-0b1c-6d7e8f9a0b1c
let
	Random.seed!(42)
	n_r = 300
	shots_r    = rand(3:18, n_r)
	poss_r     = rand(35.0:0.1:65.0, n_r)
	team_qual  = rand([0.8, 1.0, 1.2], n_r)
	goals_r    = rand.(NegativeBinomial.(2, 2 ./ (2 .+ shots_r .* 0.12 .* team_qual)))

	df_r = DataFrame(shots=shots_r, possession=poss_r, goals=goals_r)

	pois_r = glm(@formula(goals ~ shots + possession), df_r, Poisson(), LogLink())
	nb_r   = glm(@formula(goals ~ shots + possession), df_r, NegativeBinomial(), LogLink())

	DataFrame(
		modèle      = ["Poisson", "Binomiale négative"],
		AIC         = round.([aic(pois_r), aic(nb_r)]; digits=1),
		variance_moyenne = [round(var(goals_r)/mean(goals_r); digits=2), ""],
	)
end

# ╔═╡ a3b4c5d6-0021-4e2f-0b1c-6d7e8f9a0b1c
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Compris la surdispersion et ses causes
2. Implémenté la régression binomiale négative
3. Détecté et visualisé l'inflation de zéros
4. Construit un modèle hurdle en deux étapes
5. Effectué un test d'adéquation (goodness-of-fit)
6. Appliqué les critères de sélection de modèle

## Points clés

- **Poisson standard** suppose variance = moyenne (souvent violé en football)
- **Binomiale négative** gère la surdispersion (variance > moyenne)
- **L'inflation de zéros** est courante dans le football (matchs défensifs)
- **Les modèles hurdle** séparent « y aura-t-il des buts ? » de « combien ? »
- **Toujours vérifier les hypothèses** avant de choisir un modèle
- **L'AIC** aide à comparer les modèles non emboîtés
- **Les vraies données de football** nécessitent souvent la binomiale négative

## Exercices

1. **Distribution de Poisson** — tracer la distribution de Poisson pour
   différentes valeurs de lambda.
2. **Comparer tous les modèles** — ajuster Poisson, NB, ZIP et Hurdle sur
   les mêmes données.
3. **Simuler des données** — générer des données avec des propriétés
   connues et vérifier la sélection du modèle.
4. **Données réelles** — appliquer aux données réelles de championnat et
   identifier le meilleur modèle.
5. **Intervalles de prédiction** — générer des intervalles pour les
   modèles de comptage.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
Distributions = "0.25"
GLM = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─a3b4c5d6-0001-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0002-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0003-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0004-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0005-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0006-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0007-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0008-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0009-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0010-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0011-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0012-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0013-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0014-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0015-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0016-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0017-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0018-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0019-4e2f-0b1c-6d7e8f9a0b1c
# ╠═a3b4c5d6-0020-4e2f-0b1c-6d7e8f9a0b1c
# ╟─a3b4c5d6-0021-4e2f-0b1c-6d7e8f9a0b1c
# ╠═00000000-0000-0000-0000-000000000001
