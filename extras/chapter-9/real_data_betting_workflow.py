from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "Companion-Code" / "extras" / "chapter-9" / "data" / "E0_2025_2026.csv"
RESULTS_DIR = ROOT / "Companion-Code" / "extras" / "chapter-9" / "results"
IMAGES_DIR = ROOT / "Book" / "images"

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


def load_matches() -> pd.DataFrame:
    matches = pd.read_csv(DATA_PATH)
    matches["Kickoff"] = pd.to_datetime(
        matches["Date"] + " " + matches["Time"], format="%d/%m/%Y %H:%M"
    )
    matches = matches.sort_values("Kickoff").reset_index(drop=True)

    for column in ["FTHG", "FTAG", "B365H", "B365D", "B365A"]:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

    matches = matches.dropna(subset=["Kickoff", "HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"])

    for outcome in ["H", "D", "A"]:
        matches[f"Imp{outcome}"] = 1 / matches[f"B365{outcome}"]

    matches["Overround"] = matches[["ImpH", "ImpD", "ImpA"]].sum(axis=1)
    return matches


def settle_bet(result: str, prediction: str, odds: float, stake: float) -> float:
    return stake * (odds - 1) if result == prediction else -stake


def cumulative_roi(profits: list[float], stake: float) -> list[float]:
    running_profit = 0.0
    path: list[float] = []
    for index, profit in enumerate(profits, start=1):
        running_profit += profit
        path.append(running_profit / (index * stake))
    return path


def baseline_paths(matches: pd.DataFrame, stake: float = 100.0) -> dict[str, dict[str, object]]:
    home_profits = [
        settle_bet(row.FTR, "H", row.B365H, stake)
        for row in matches.itertuples(index=False)
    ]

    favorite_profits = []
    favorite_predictions = []
    outcome_map = {"B365H": "H", "B365D": "D", "B365A": "A"}
    for row in matches.itertuples(index=False):
        best_column = min(["B365H", "B365D", "B365A"], key=lambda column: getattr(row, column))
        favorite_predictions.append(outcome_map[best_column])
        favorite_profits.append(
            settle_bet(row.FTR, outcome_map[best_column], getattr(row, best_column), stake)
        )

    return {
        "home": {
            "predictions": ["H"] * len(matches),
            "profits": home_profits,
            "roi_path": cumulative_roi(home_profits, stake),
            "accuracy": (matches["FTR"] == "H").mean(),
            "profit": float(sum(home_profits)),
            "roi": float(sum(home_profits) / (len(matches) * stake)),
        },
        "favorite": {
            "predictions": favorite_predictions,
            "profits": favorite_profits,
            "roi_path": cumulative_roi(favorite_profits, stake),
            "accuracy": sum(
                prediction == actual
                for prediction, actual in zip(favorite_predictions, matches["FTR"])
            )
            / len(matches),
            "profit": float(sum(favorite_profits)),
            "roi": float(sum(favorite_profits) / (len(matches) * stake)),
        },
    }


def train_validation_split(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(matches) * 2 / 3)
    return matches.iloc[:cutoff].copy(), matches.iloc[cutoff:].copy()


def build_calibrated_model(train_df: pd.DataFrame) -> tuple[Pipeline, CalibratedClassifierCV]:
    features = ["HomeTeam", "AwayTeam", "ImpH", "ImpD", "ImpA", "Overround"]
    preprocess = ColumnTransformer(
        [
            ("teams", OneHotEncoder(handle_unknown="ignore"), ["HomeTeam", "AwayTeam"]),
            ("numeric", StandardScaler(), ["ImpH", "ImpD", "ImpA", "Overround"]),
        ]
    )

    base_model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    base_model.fit(train_df[features], train_df["FTR"])

    calibrated = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    calibrated.fit(train_df[features], train_df["FTR"])
    return base_model, calibrated


def evaluate_model(
    base_model: Pipeline,
    calibrated_model: CalibratedClassifierCV,
    valid_df: pd.DataFrame,
    stake: float = 100.0,
    starting_bankroll: float = 10_000.0,
) -> dict[str, object]:
    features = ["HomeTeam", "AwayTeam", "ImpH", "ImpD", "ImpA", "Overround"]

    base_probs = base_model.predict_proba(valid_df[features])
    base_predictions = base_model.predict(valid_df[features])

    calibrated_probs = calibrated_model.predict_proba(valid_df[features])
    calibrated_predictions = calibrated_model.predict(valid_df[features])
    classes = list(calibrated_model.classes_)
    class_index = {label: index for index, label in enumerate(classes)}

    flat_profit_path: list[float] = []
    flat_running_profit = 0.0
    flat_bets = 0
    flat_wins = 0
    flat_roi_path: list[float] = []

    kelly_bankroll = starting_bankroll
    kelly_roi_path: list[float] = []
    kelly_bets = 0

    for row_index, row in enumerate(valid_df.itertuples(index=False)):
        best_bet: dict[str, float | str] | None = None
        for outcome, odds_column in [("H", "B365H"), ("D", "B365D"), ("A", "B365A")]:
            odds = getattr(row, odds_column)
            probability = calibrated_probs[row_index, class_index[outcome]]
            expected_value = probability * odds - 1
            candidate = {
                "outcome": outcome,
                "odds": odds,
                "probability": probability,
                "expected_value": expected_value,
            }
            if best_bet is None or candidate["expected_value"] > best_bet["expected_value"]:
                best_bet = candidate

        assert best_bet is not None

        if best_bet["expected_value"] > 0:
            flat_bets += 1
            flat_profit = settle_bet(
                row.FTR,
                best_bet["outcome"],  # type: ignore[arg-type]
                float(best_bet["odds"]),
                stake,
            )
            flat_running_profit += flat_profit
            flat_wins += int(row.FTR == best_bet["outcome"])

            net_odds = float(best_bet["odds"]) - 1
            probability = float(best_bet["probability"])
            kelly_fraction = max((net_odds * probability - (1 - probability)) / net_odds, 0.0)
            kelly_stake = min(kelly_bankroll * kelly_fraction, kelly_bankroll)
            if kelly_stake > 0:
                kelly_bets += 1
                kelly_bankroll -= kelly_stake
                if row.FTR == best_bet["outcome"]:
                    kelly_bankroll += kelly_stake * float(best_bet["odds"])

        flat_roi_path.append(flat_running_profit / (flat_bets * stake) if flat_bets else 0.0)
        kelly_roi_path.append((kelly_bankroll - starting_bankroll) / starting_bankroll)

    return {
        "base_accuracy": float(accuracy_score(valid_df["FTR"], base_predictions)),
        "base_log_loss": float(log_loss(valid_df["FTR"], base_probs, labels=base_model.classes_)),
        "calibrated_accuracy": float(accuracy_score(valid_df["FTR"], calibrated_predictions)),
        "calibrated_log_loss": float(
            log_loss(valid_df["FTR"], calibrated_probs, labels=calibrated_model.classes_)
        ),
        "flat_value_bets": flat_bets,
        "flat_value_win_rate": float(flat_wins / flat_bets) if flat_bets else 0.0,
        "flat_value_profit": float(flat_running_profit),
        "flat_value_roi": float(flat_running_profit / (flat_bets * stake)) if flat_bets else 0.0,
        "flat_roi_path": flat_roi_path,
        "kelly_bets": kelly_bets,
        "kelly_final_bankroll": float(kelly_bankroll),
        "kelly_profit": float(kelly_bankroll - starting_bankroll),
        "kelly_roi": float((kelly_bankroll - starting_bankroll) / starting_bankroll),
        "kelly_roi_path": kelly_roi_path,
        "classes": classes,
        "calibrated_probabilities": calibrated_probs,
    }


def plot_roi_series(kickoffs: pd.Series, roi_path: list[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(kickoffs, roi_path, linewidth=2, color="#0B5D7A")
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Kickoff Date")
    plt.ylabel("ROI")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_model_roi(
    kickoffs: pd.Series,
    flat_roi_path: list[float],
    kelly_roi_path: list[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(kickoffs, flat_roi_path, label="Flat Stake Value Bets", linewidth=2, color="#0B5D7A")
    plt.plot(kickoffs, kelly_roi_path, label="Full Kelly", linewidth=2, color="#B63A1B")
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.title("Validation ROI Over Time for the Calibrated Betting Model")
    plt.xlabel("Kickoff Date")
    plt.ylabel("ROI")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_calibration(valid_df: pd.DataFrame, calibrated_probabilities, classes: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, label in zip(axes, classes):
        y_true = (valid_df["FTR"] == label).astype(int)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            calibrated_probabilities[:, classes.index(label)],
            n_bins=6,
            strategy="quantile",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        axis.plot(mean_predicted_value, fraction_of_positives, marker="o", linewidth=2, color="#0B5D7A")
        axis.set_title(f"{label} Calibration")
        axis.set_xlabel("Predicted Probability")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Observed Frequency")
    fig.suptitle("Validation Calibration Curves for Home, Draw, and Away Outcomes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_preview(matches: pd.DataFrame) -> None:
    preview = matches[MIN_COLUMNS].head(5)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preview.to_csv(RESULTS_DIR / "chapter9_minimal_preview.csv", index=False)


def write_summary(matches: pd.DataFrame, baseline_summary: dict[str, dict[str, object]], model_summary: dict[str, object], train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    baseline_headlines = {}
    for name, summary in baseline_summary.items():
        baseline_headlines[name] = {
            "accuracy": summary["accuracy"],
            "profit": summary["profit"],
            "roi": summary["roi"],
        }

    summary = {
        "matches": int(len(matches)),
        "date_min": str(matches["Kickoff"].min()),
        "date_max": str(matches["Kickoff"].max()),
        "columns_used": MIN_COLUMNS,
        "average_overround": float(matches["Overround"].mean()),
        "train_matches": int(len(train_df)),
        "validation_matches": int(len(valid_df)),
        "train_start": str(train_df["Kickoff"].min()),
        "train_end": str(train_df["Kickoff"].max()),
        "validation_start": str(valid_df["Kickoff"].min()),
        "validation_end": str(valid_df["Kickoff"].max()),
        "home_strategy": baseline_headlines["home"],
        "favorite_strategy": baseline_headlines["favorite"],
        "model": {
            key: value
            for key, value in model_summary.items()
            if key not in {"calibrated_probabilities", "classes", "flat_roi_path", "kelly_roi_path"}
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "chapter9_real_data_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    save_preview(matches)

    baselines = baseline_paths(matches)
    train_df, valid_df = train_validation_split(matches)
    base_model, calibrated_model = build_calibrated_model(train_df)
    model_summary = evaluate_model(base_model, calibrated_model, valid_df)

    plot_roi_series(
        matches["Kickoff"],
        baselines["home"]["roi_path"],  # type: ignore[arg-type]
        "ROI Over Time for Always Betting the Home Team",
        IMAGES_DIR / "ch09_home_roi_actual.png",
    )
    plot_roi_series(
        matches["Kickoff"],
        baselines["favorite"]["roi_path"],  # type: ignore[arg-type]
        "ROI Over Time for Always Betting the Lowest Bet365 Odds",
        IMAGES_DIR / "ch09_favorite_roi_actual.png",
    )
    plot_model_roi(
        valid_df["Kickoff"],
        model_summary["flat_roi_path"],  # type: ignore[arg-type]
        model_summary["kelly_roi_path"],  # type: ignore[arg-type]
        IMAGES_DIR / "ch09_model_value_roi_actual.png",
    )
    plot_calibration(
        valid_df,
        model_summary["calibrated_probabilities"],
        model_summary["classes"],  # type: ignore[arg-type]
        IMAGES_DIR / "ch09_model_calibration_actual.png",
    )

    write_summary(matches, baselines, model_summary, train_df, valid_df)


if __name__ == "__main__":
    main()
