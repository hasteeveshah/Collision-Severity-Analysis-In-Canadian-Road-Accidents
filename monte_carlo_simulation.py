from pathlib import Path

import numpy as np
import pandas as pd

import simulation_analysis as sim

OUTPUT_DIR = Path("outputs/monte_carlo")
N_MONTE_CARLO = 5000


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_monte_carlo_profiles(
    df_training: pd.DataFrame,
    features: list[str],
    *,
    n_draws: int = N_MONTE_CARLO,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap realistic profiles from the observed feature distribution.
    """
    return df_training[features].sample(n=n_draws, replace=True, random_state=random_state).reset_index(drop=True)


def predict_profiles(
    res,
    profiles: pd.DataFrame,
    df_training: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
    *,
    include_intervals: bool = True,
) -> pd.DataFrame:
    """
    Score one or more profiles using whichever prediction API is available in
    simulation_analysis.py.
    """
    if hasattr(sim, "predict_profile_probabilities"):
        return sim.predict_profile_probabilities(
            res,
            profiles,
            df_training,
            features,
            model_columns,
            include_intervals=include_intervals,
        )

    probabilities = []
    for idx in range(len(profiles)):
        profile = profiles.iloc[[idx]].copy()
        p = sim.predict_probability(res, profile, df_training, features, model_columns)
        probabilities.append(float(p))

    return pd.DataFrame(
        {
            "Predicted_Probability": probabilities,
            "CI_Lower_95": np.nan,
            "CI_Upper_95": np.nan,
        }
    )


def fit_simulation_model():
    """
    Normalize the simulation fit output across different simulation_analysis versions.
    """
    fit_output = sim.fit_combined_binary_model()
    if len(fit_output) == 6:
        res, df_c, features, categorical_cols, model_columns, fit_type = fit_output
    else:
        res, df_c, features, categorical_cols, model_columns = fit_output
        fit_type = "Unknown"

    return res, df_c, features, categorical_cols, model_columns, fit_type


def summarize_probability_distribution(probabilities: pd.Series) -> pd.DataFrame:
    """
    Provide a concise summary for the Monte Carlo predicted-risk distribution.
    """
    summary = {
        "n_draws": int(probabilities.shape[0]),
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(probabilities.median()),
        "std_probability": float(probabilities.std(ddof=1)),
        "min_probability": float(probabilities.min()),
        "p05_probability": float(probabilities.quantile(0.05)),
        "p25_probability": float(probabilities.quantile(0.25)),
        "p75_probability": float(probabilities.quantile(0.75)),
        "p95_probability": float(probabilities.quantile(0.95)),
        "max_probability": float(probabilities.max()),
    }
    return pd.DataFrame([summary])


def summarize_risk_bands(probabilities: pd.Series) -> pd.DataFrame:
    """
    Group Monte Carlo probabilities into a few easy-to-read risk bands.
    """
    bands = [
        ("Below 0.25", probabilities < 0.25),
        ("0.25 to <0.50", (probabilities >= 0.25) & (probabilities < 0.50)),
        ("0.50 to <0.75", (probabilities >= 0.50) & (probabilities < 0.75)),
        ("0.75 and above", probabilities >= 0.75),
    ]

    records = []
    total = len(probabilities)
    for label, mask in bands:
        count = int(mask.sum())
        records.append(
            {
                "Risk_Band": label,
                "Count": count,
                "Share": float(count / total) if total else np.nan,
            }
        )

    return pd.DataFrame(records)


def build_brief_details(
    baseline_profile: pd.DataFrame,
    baseline_prediction: pd.Series,
    summary: pd.DataFrame,
    risk_bands: pd.DataFrame,
) -> list[str]:
    """
    Create a few concise text takeaways from the Monte Carlo run.
    """
    summary_row = summary.iloc[0]
    high_risk_share = risk_bands.loc[risk_bands["Risk_Band"] == "0.75 and above", "Share"].iloc[0]

    details = [
        (
            "Baseline profile: "
            f"C_WTHR={baseline_profile.iloc[0]['C_WTHR']}, "
            f"C_RSUR={baseline_profile.iloc[0]['C_RSUR']}, "
            f"C_RALN={baseline_profile.iloc[0]['C_RALN']}, "
            f"C_TRAF={baseline_profile.iloc[0]['C_TRAF']}, "
            f"P_SAFE={baseline_profile.iloc[0]['P_SAFE']}, "
            f"P_USER={baseline_profile.iloc[0]['P_USER']}, "
            f"P_SEX={baseline_profile.iloc[0]['P_SEX']}, "
            f"V_TYPE={baseline_profile.iloc[0]['V_TYPE']}, "
            f"VEHICLE_AGE={baseline_profile.iloc[0]['VEHICLE_AGE']}, "
            f"P_AGE={baseline_profile.iloc[0]['P_AGE']}"
        ),
        (
            "Baseline predicted probability: "
            f"{float(baseline_prediction['Predicted_Probability']):.4f} "
            f"(95% CI {float(baseline_prediction['CI_Lower_95']):.4f} to "
            f"{float(baseline_prediction['CI_Upper_95']):.4f})"
        ),
        (
            "Monte Carlo central range: "
            f"5th percentile {float(summary_row['p05_probability']):.4f}, "
            f"median {float(summary_row['median_probability']):.4f}, "
            f"95th percentile {float(summary_row['p95_probability']):.4f}"
        ),
        f"High-risk draws (>= 0.75): {float(high_risk_share):.1%}",
    ]
    return details


def run_monte_carlo_simulation(
    *,
    n_draws: int = N_MONTE_CARLO,
    random_state: int = 42,
    save_outputs: bool = True,
) -> dict[str, object]:
    """
    Sample observed feature profiles, score them with the combined binary model,
    and save both the draws and a summary table.
    """
    res, df_c, features, categorical_cols, model_columns, fit_type = fit_simulation_model()
    baseline = sim.make_baseline_profile(df_c, features)
    baseline_pred = predict_profiles(
        res,
        baseline,
        df_c,
        features,
        model_columns,
        include_intervals=True,
    ).iloc[0]

    draws = build_monte_carlo_profiles(df_c, features, n_draws=n_draws, random_state=random_state)
    draw_predictions = predict_profiles(
        res,
        draws,
        df_c,
        features,
        model_columns,
        include_intervals=False,
    )
    monte_carlo_draws = pd.concat([draws.reset_index(drop=True), draw_predictions.reset_index(drop=True)], axis=1)
    monte_carlo_draws["Fit_Type"] = fit_type

    summary = summarize_probability_distribution(monte_carlo_draws["Predicted_Probability"])
    risk_bands = summarize_risk_bands(monte_carlo_draws["Predicted_Probability"])
    summary["Baseline_Probability"] = float(baseline_pred["Predicted_Probability"])
    summary["Baseline_CI_Lower_95"] = float(baseline_pred["CI_Lower_95"]) if pd.notna(baseline_pred["CI_Lower_95"]) else np.nan
    summary["Baseline_CI_Upper_95"] = float(baseline_pred["CI_Upper_95"]) if pd.notna(baseline_pred["CI_Upper_95"]) else np.nan
    brief_details = build_brief_details(baseline, baseline_pred, summary, risk_bands)

    outputs: dict[str, object] = {
        "fit_result": res,
        "training_df": df_c,
        "features": features,
        "categorical_cols": categorical_cols,
        "model_columns": model_columns,
        "baseline_profile": baseline,
        "baseline_prediction": baseline_pred,
        "monte_carlo_draws": monte_carlo_draws,
        "monte_carlo_summary": summary,
        "risk_bands": risk_bands,
        "brief_details": brief_details,
    }

    if save_outputs:
        ensure_output_dir()
        monte_carlo_draws.to_csv(OUTPUT_DIR / "monte_carlo_draws.csv", index=False)
        summary.to_csv(OUTPUT_DIR / "monte_carlo_summary.csv", index=False)
        risk_bands.to_csv(OUTPUT_DIR / "monte_carlo_risk_bands.csv", index=False)

    return outputs


def main() -> None:
    outputs = run_monte_carlo_simulation(save_outputs=True)
    print("\nBrief Monte Carlo details:")
    for detail in outputs["brief_details"]:
        print(f"- {detail}")
    print("\nMonte Carlo summary:")
    print(outputs["monte_carlo_summary"].round(4))
    print("\nRisk bands:")
    print(outputs["risk_bands"].round(4))
    print(f"\nSaved Monte Carlo outputs to: {ensure_output_dir().resolve()}")


if __name__ == "__main__":
    main()
