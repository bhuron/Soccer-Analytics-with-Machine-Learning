from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
MIN_COLUMNS = [
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_matches(data_url: str = DATA_URL) -> pd.DataFrame:
    matches = pd.read_csv(data_url)
    matches = matches[MIN_COLUMNS].dropna().copy()

    matches["Kickoff"] = pd.to_datetime(
        matches["Date"] + " " + matches["Time"],
        format="%d/%m/%Y %H:%M",
    )
    matches = matches.sort_values("Kickoff").reset_index(drop=True)

    for outcome in ["H", "D", "A"]:
        matches[f"Imp{outcome}"] = 1 / matches[f"B365{outcome}"]
    matches["Overround"] = matches[["ImpH", "ImpD", "ImpA"]].sum(axis=1)

    odds_to_result = {"B365H": "H", "B365D": "D", "B365A": "A"}
    matches["home_pick"] = "H"
    matches["lowest_odds_pick"] = (
        matches[["B365H", "B365D", "B365A"]]
        .idxmin(axis=1)
        .map(odds_to_result)
    )
    return matches


def train_validation_split(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(matches) * 2 / 3)
    return matches.iloc[:cutoff].copy(), matches.iloc[cutoff:].copy()


def confusion_table(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    return pd.crosstab(
        actual,
        predicted,
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )


def build_ml_model(train_df: pd.DataFrame) -> Pipeline:
    features = ["HomeTeam", "AwayTeam", "ImpH", "ImpD", "ImpA", "Overround"]
    preprocess = ColumnTransformer(
        [
            ("teams", OneHotEncoder(handle_unknown="ignore"), ["HomeTeam", "AwayTeam"]),
            ("numeric", StandardScaler(), ["ImpH", "ImpD", "ImpA", "Overround"]),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(train_df[features], train_df["FTR"])
    return model


def add_ml_predictions(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    model = build_ml_model(train_df)
    features = ["HomeTeam", "AwayTeam", "ImpH", "ImpD", "ImpA", "Overround"]
    valid = valid_df.copy()
    valid["ml_pick"] = model.predict(valid[features])
    return model, valid


def accuracy_summary(valid_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "strategy": ["always_home", "lowest_odds", "machine_learning"],
            "accuracy": [
                (valid_df["home_pick"] == valid_df["FTR"]).mean(),
                (valid_df["lowest_odds_pick"] == valid_df["FTR"]).mean(),
                (valid_df["ml_pick"] == valid_df["FTR"]).mean(),
            ],
        }
    )
    return summary.sort_values("accuracy", ascending=False).reset_index(drop=True)


def fixed_stake_summary(matches: pd.DataFrame, picks: pd.Series, stake: float = 100.0) -> tuple[float, float]:
    total_profit = 0.0
    for pick, (_, row) in zip(picks, matches.iterrows()):
        odds = row[f"B365{pick}"]
        profit = stake * (odds - 1) if row["FTR"] == pick else -stake
        total_profit += profit

    total_staked = stake * len(matches)
    roi = total_profit / total_staked if total_staked else 0.0
    return float(total_profit), float(roi)


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    q = 1 - probability
    if b <= 0:
        return 0.0
    return max((b * probability - q) / b, 0.0)


def kelly_summary(
    matches: pd.DataFrame,
    picks: pd.Series,
    probabilities: list[float],
    starting_bankroll: float = 1000.0,
) -> tuple[float, float]:
    bankroll = starting_bankroll

    for pick, probability, (_, row) in zip(picks, probabilities, matches.iterrows()):
        odds = row[f"B365{pick}"]
        fraction = min(kelly_fraction(probability, odds), 1.0)
        stake = bankroll * fraction
        bankroll -= stake

        if row["FTR"] == pick:
            bankroll += stake * odds

    roi = (bankroll - starting_bankroll) / starting_bankroll if starting_bankroll else 0.0
    return float(bankroll), float(roi)


def build_results() -> dict[str, object]:
    matches = load_matches()
    train_df, valid_df = train_validation_split(matches)
    _, valid_df = add_ml_predictions(train_df, valid_df)

    comparison = accuracy_summary(valid_df)

    home_confusion = confusion_table(valid_df["FTR"], valid_df["home_pick"])
    lowest_confusion = confusion_table(valid_df["FTR"], valid_df["lowest_odds_pick"])
    ml_confusion = confusion_table(valid_df["FTR"], valid_df["ml_pick"])

    home_fixed_profit, home_fixed_roi = fixed_stake_summary(valid_df, valid_df["home_pick"])
    lowest_fixed_profit, lowest_fixed_roi = fixed_stake_summary(valid_df, valid_df["lowest_odds_pick"])

    home_rule_probability = float((train_df["FTR"] == "H").mean())
    lowest_rule_probability = float((train_df["lowest_odds_pick"] == train_df["FTR"]).mean())

    home_kelly_bankroll, home_kelly_roi = kelly_summary(
        valid_df,
        valid_df["home_pick"],
        [home_rule_probability] * len(valid_df),
    )
    lowest_kelly_bankroll, lowest_kelly_roi = kelly_summary(
        valid_df,
        valid_df["lowest_odds_pick"],
        [lowest_rule_probability] * len(valid_df),
    )

    return {
        "matches": matches,
        "train_df": train_df,
        "valid_df": valid_df,
        "comparison": comparison,
        "home_confusion": home_confusion,
        "lowest_confusion": lowest_confusion,
        "ml_confusion": ml_confusion,
        "home_fixed_profit": home_fixed_profit,
        "home_fixed_roi": home_fixed_roi,
        "lowest_fixed_profit": lowest_fixed_profit,
        "lowest_fixed_roi": lowest_fixed_roi,
        "home_rule_probability": home_rule_probability,
        "lowest_rule_probability": lowest_rule_probability,
        "home_kelly_bankroll": home_kelly_bankroll,
        "home_kelly_roi": home_kelly_roi,
        "lowest_kelly_bankroll": lowest_kelly_bankroll,
        "lowest_kelly_roi": lowest_kelly_roi,
    }


def write_outputs(results: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    matches = results["matches"]
    comparison = results["comparison"]
    preview = matches[MIN_COLUMNS].head(10)
    preview.to_csv(RESULTS_DIR / "chapter9_minimal_preview.csv", index=False)

    summary = {
        "matches": int(len(matches)),
        "date_min": str(matches["Kickoff"].min()),
        "date_max": str(matches["Kickoff"].max()),
        "columns_used": MIN_COLUMNS,
        "train_matches": int(len(results["train_df"])),
        "validation_matches": int(len(results["valid_df"])),
        "accuracy_ranking": comparison.to_dict(orient="records"),
        "fixed_stake": {
            "always_home": {
                "profit": results["home_fixed_profit"],
                "roi": results["home_fixed_roi"],
            },
            "lowest_odds": {
                "profit": results["lowest_fixed_profit"],
                "roi": results["lowest_fixed_roi"],
            },
        },
        "kelly": {
            "always_home": {
                "bankroll": results["home_kelly_bankroll"],
                "roi": results["home_kelly_roi"],
            },
            "lowest_odds": {
                "bankroll": results["lowest_kelly_bankroll"],
                "roi": results["lowest_kelly_roi"],
            },
        },
    }
    with (RESULTS_DIR / "chapter9_real_data_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    results = build_results()
    write_outputs(results)
    print(results["comparison"])


if __name__ == "__main__":
    main()
