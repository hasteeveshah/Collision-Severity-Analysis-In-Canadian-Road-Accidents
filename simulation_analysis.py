from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from Load_prepare import df, RANDOM_STATE, N_SAMPLE_MAIN, MIN_COUNT_RARE
from models import preprocess_categoricals_variablewise
from utils import build_design_matrix, make_or_table, reduce_to_full_rank

SIMULATION_CATEGORICAL_COLS = ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF", "P_SAFE", "P_USER", "P_SEX", "V_TYPE"]
OUTPUT_DIR = Path("outputs/simulation")
DEFAULT_BASELINE_PROFILE = {
    "C_WTHR": "1",
    "C_RSUR": "1",
    "C_RALN": "1",
    "C_TRAF": "01",
    "C_MNTH": 7.0,
    "C_WDAY": 4.0,
    "C_HOUR": 14.0,
    "C_VEHS": 2.0,
    "P_SAFE": "02",
    "P_USER": "1",
    "P_SEX": "F",
    "V_TYPE": "01",
    "VEHICLE_AGE": 7.0,
    "P_AGE": 34.0,
}

def _is_binary_dummy(series: pd.Series) -> bool:
    non_na = pd.Series(series).dropna()
    if non_na.empty:
        return False
    unique_vals = pd.Index(non_na.unique())
    return bool(unique_vals.isin([0, 1, 0.0, 1.0, False, True]).all())


def _drop_sparse_or_separating_dummies(
    X_local: pd.DataFrame,
    y_local: pd.Series,
) -> pd.DataFrame:
    min_active_support = max(8, int(0.0005 * len(X_local)))
    min_active_class_support = max(2, int(0.00005 * len(X_local)))

    keep_cols: list[str] = []
    for col in X_local.columns:
        if col == "const":
            keep_cols.append(col)
            continue

        col_series = X_local[col]
        if not _is_binary_dummy(col_series):
            keep_cols.append(col)
            continue

        active_mask = col_series.astype("float32") > 0.5
        active_count = int(active_mask.sum())
        inactive_count = int((~active_mask).sum())

        if active_count < min_active_support or inactive_count == 0:
            continue

        active_y = y_local.loc[active_mask]
        active_class_counts = active_y.value_counts()
        if len(active_class_counts) < 2:
            continue
        if int(active_class_counts.min()) < min_active_class_support:
            continue

        keep_cols.append(col)

    if not keep_cols:
        keep_cols = ["const"] if "const" in X_local.columns else []

    return X_local[keep_cols].copy()


def prepare_combined_binary_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare the same combined-model dataset used by the binary analysis,
    using the combined model with the grouped OTHER bucket.
    """
    df_person = df[df["P_ISEV"].isin([1, 2, 3])].copy()
    df_person["Y_INJURY"] = df_person["P_ISEV"].isin([2, 3]).astype(int)

    rq1_features = ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF", "C_MNTH", "C_WDAY", "C_HOUR", "C_VEHS"]
    rq2_features = ["P_SAFE", "P_USER", "P_SEX", "V_TYPE", "VEHICLE_AGE", "P_AGE"]
    combined_features = rq1_features + rq2_features

    categorical_cols = SIMULATION_CATEGORICAL_COLS.copy()

    df_c = df_person.dropna(subset=combined_features + ["Y_INJURY"]).copy()
    df_c = preprocess_categoricals_variablewise(
        df_c,
        categorical_cols,
        include_other_group=True,
        default_rare_threshold=max(MIN_COUNT_RARE, 5000),
        default_keep_levels={"OTHER"},
    )

    # Same sample cap as your binary pipeline
    df_c = df_c.sample(min(N_SAMPLE_MAIN, len(df_c)), random_state=RANDOM_STATE).copy()

    return df_c, combined_features, categorical_cols


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def fit_combined_binary_model():
    """
    Fit the final combined binary model and return:
    - fitted result
    - modeling dataframe
    - feature list
    - design matrix columns
    - fit type
    """
    df_c, combined_features, categorical_cols = prepare_combined_binary_data()

    y = df_c["Y_INJURY"].astype(int)
    X = build_design_matrix(df_c, combined_features, drop_first=True, add_constant=True)

    # safety cleanup mirroring models.py
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.loc[:, ~X.T.duplicated()]
    X = _drop_sparse_or_separating_dummies(X, y)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.loc[:, ~X.T.duplicated()]
    X, _ = reduce_to_full_rank(X)

    model = sm.Logit(y, X)
    res = None
    last_mle_result = None

    for method_name, iter_budget in [("newton", 300), ("bfgs", 800), ("lbfgs", 800)]:
        try:
            candidate = model.fit(method=method_name, maxiter=iter_budget, disp=0)
            mle_retvals = getattr(candidate, "mle_retvals", {}) or {}
            converged = bool(mle_retvals.get("converged", getattr(candidate, "converged", True)))
            last_mle_result = candidate
            if converged:
                res = candidate
                break
        except Exception:
            continue

    if res is None and last_mle_result is not None:
        res = last_mle_result
        fit_type = "MLE"
    elif res is not None:
        fit_type = "MLE"
    else:
        print("\nCombined binary MLE failed during simulation build. Switching to regularized fit.")
        res = model.fit_regularized(method="l1", alpha=0.01, maxiter=5000)
        fit_type = "Regularized (L1)"

    print(f"\nSimulation model fitted using: {fit_type}")
    return res, df_c, combined_features, categorical_cols, X.columns.tolist(), fit_type


def make_baseline_profile(df_c: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Create a typical baseline profile:
    - mode for categoricals
    - median for numerics
    """
    row = {}

    for col in features:
        if col in DEFAULT_BASELINE_PROFILE:
            row[col] = DEFAULT_BASELINE_PROFILE[col]
        elif col in SIMULATION_CATEGORICAL_COLS:
            row[col] = df_c[col].mode(dropna=True).iloc[0]
        else:
            row[col] = float(df_c[col].median())

    return pd.DataFrame([row])


def align_design_matrix(
    df_profiles: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
) -> pd.DataFrame:
    """
    Convert scenario profiles to the exact same dummy-encoded matrix as the trained model.
    """
    profiles = df_profiles[features].copy()

    # Reuse the training category space so we can encode new rows directly
    # instead of rebuilding a training-sized matrix for every prediction.
    for col in features:
        train_col = df_training_features[col]
        if col in SIMULATION_CATEGORICAL_COLS:
            categories = pd.Index(sorted(train_col.astype("string").dropna().unique().tolist()))
            profiles[col] = pd.Categorical(profiles[col].astype("string"), categories=categories)
        else:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce")

    X_new = pd.get_dummies(profiles[features], drop_first=True)
    X_new = X_new.apply(pd.to_numeric, errors="coerce").astype("float32")
    X_new = sm.add_constant(X_new, has_constant="add")

    # Add missing columns from training matrix
    for col in model_columns:
        if col not in X_new.columns:
            X_new[col] = 0.0

    # Drop unexpected columns
    X_new = X_new[[col for col in model_columns]]

    return X_new.astype("float32")


def predict_probability(
    res,
    profile_df: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
) -> float:
    """
    Predict injury probability for a single profile.
    """
    predictions = predict_profile_probabilities(
        res,
        profile_df,
        df_training_features,
        features,
        model_columns,
    )
    return float(predictions.iloc[0]["Predicted_Probability"])


def _compute_prediction_intervals(
    res,
    X_new: pd.DataFrame,
    *,
    z_value: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Wald-style 95% confidence intervals on the probability scale.
    """
    probs = pd.Series(res.predict(X_new), index=X_new.index, dtype="float64").to_numpy()

    try:
        cov_params = np.asarray(res.cov_params(), dtype="float64")
        beta = np.asarray(res.params, dtype="float64")
    except Exception:
        nan = np.full(len(X_new), np.nan, dtype="float64")
        return probs, nan, nan

    x_mat = X_new.to_numpy(dtype="float64")
    eta = x_mat @ beta
    eta_var = np.einsum("ij,jk,ik->i", x_mat, cov_params, x_mat)
    eta_se = np.sqrt(np.clip(eta_var, a_min=0.0, a_max=None))

    lower = 1.0 / (1.0 + np.exp(-(eta - z_value * eta_se)))
    upper = 1.0 / (1.0 + np.exp(-(eta + z_value * eta_se)))

    return probs, lower, upper


def predict_profile_probabilities(
    res,
    profiles_df: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
    *,
    include_intervals: bool = True,
) -> pd.DataFrame:
    """
    Predict many profiles in one vectorized call.
    """
    X_new = align_design_matrix(profiles_df, df_training_features, features, model_columns)
    if include_intervals:
        probs, lower, upper = _compute_prediction_intervals(res, X_new)
    else:
        probs = pd.Series(res.predict(X_new), index=profiles_df.index, dtype="float64").to_numpy()
        lower = np.full(len(profiles_df), np.nan, dtype="float64")
        upper = np.full(len(profiles_df), np.nan, dtype="float64")

    return pd.DataFrame(
        {
            "Predicted_Probability": probs,
            "CI_Lower_95": lower,
            "CI_Upper_95": upper,
        },
        index=profiles_df.index,
    )


def score_profiles(
    res,
    profiles_df: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
) -> pd.DataFrame:
    """
    Return the original profile columns with prediction intervals attached.
    """
    predictions = predict_profile_probabilities(res, profiles_df, df_training_features, features, model_columns)
    return pd.concat([profiles_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)


def run_category_switch_simulations(
    res,
    baseline: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
) -> pd.DataFrame:
    """
    Switch one category at a time from baseline and compare probability.
    Edit the scenarios below if you want to use different codes.
    """
    baseline_pred = predict_profile_probabilities(res, baseline, df_training_features, features, model_columns).iloc[0]
    baseline_prob = float(baseline_pred["Predicted_Probability"])

    scenario_list = [
        ("Baseline (traffic signal)", "C_TRAF", baseline.iloc[0]["C_TRAF"]),
        ("Traffic signal removed", "C_TRAF", "18"),
        ("Weather baseline to adverse", "C_WTHR", "2"),
        ("Weather baseline to severe", "C_WTHR", "6"),
        ("Road surface baseline to risky", "C_RSUR", "6"),
        ("Road alignment baseline to risky", "C_RALN", "3"),
        ("Safety baseline to P_SAFE_09", "P_SAFE", "09"),
        ("User baseline to P_USER_5", "P_USER", "5"),
    ]

    scenario_profiles = []
    metadata = []
    for scenario_name, var, new_value in scenario_list:
        s = baseline.copy()
        s[var] = new_value
        scenario_profiles.append(s)
        metadata.append((scenario_name, var, str(baseline.iloc[0][var]), str(new_value)))

    scored = predict_profile_probabilities(
        res,
        pd.concat(scenario_profiles, ignore_index=True),
        df_training_features,
        features,
        model_columns,
    )

    records = []
    for idx, (scenario_name, var, baseline_value, scenario_value) in enumerate(metadata):
        pred = scored.iloc[idx]
        p_new = float(pred["Predicted_Probability"])
        records.append(
            {
                "Scenario": scenario_name,
                "Variable": var,
                "Baseline_Value": baseline_value,
                "Scenario_Value": scenario_value,
                "Predicted_Probability": p_new,
                "CI_Lower_95": float(pred["CI_Lower_95"]),
                "CI_Upper_95": float(pred["CI_Upper_95"]),
                "Absolute_Change": p_new - baseline_prob,
                "Relative_Change_Percent": ((p_new - baseline_prob) / baseline_prob * 100.0) if baseline_prob != 0 else np.nan,
            }
        )

    return pd.DataFrame(records)


def run_continuous_sensitivity(
    res,
    baseline: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
    variable: str,
    values: list,
) -> pd.DataFrame:
    """
    Vary one continuous variable while holding others fixed.
    """
    records = []

    for v in values:
        s = baseline.copy()
        s[variable] = v
        pred = predict_profile_probabilities(res, s, df_training_features, features, model_columns).iloc[0]

        records.append({
            "Variable": variable,
            "Value": v,
            "Predicted_Probability": float(pred["Predicted_Probability"]),
            "CI_Lower_95": float(pred["CI_Lower_95"]),
            "CI_Upper_95": float(pred["CI_Upper_95"]),
        })

    return pd.DataFrame(records)


def run_risk_profiles(
    res,
    baseline: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
) -> pd.DataFrame:
    """
    Build low / medium / high-risk scenario bundles.
    Adjust categories if you want different profiles.
    """
    profiles = []

    low = baseline.copy()
    low["VEHICLE_AGE"] = 2
    low["P_AGE"] = 25
    profiles.append(("Low-risk profile", low))

    medium = baseline.copy()
    medium["VEHICLE_AGE"] = 10
    medium["P_AGE"] = 45
    medium["C_WTHR"] = "2"
    medium["C_RALN"] = "3"
    profiles.append(("Medium-risk profile", medium))

    high = baseline.copy()
    high["VEHICLE_AGE"] = 18
    high["P_AGE"] = 70
    high["C_WTHR"] = "6"
    high["C_RSUR"] = "6"
    high["C_RALN"] = "3"
    high["C_TRAF"] = "18"
    high["P_SAFE"] = "09"
    high["P_USER"] = "5"
    profiles.append(("High-risk profile", high))

    rows = []
    scored = predict_profile_probabilities(
        res,
        pd.concat([prof for _, prof in profiles], ignore_index=True),
        df_training_features,
        features,
        model_columns,
    )
    for idx, (name, _) in enumerate(profiles):
        pred = scored.iloc[idx]
        rows.append({
            "Profile": name,
            "Predicted_Probability": float(pred["Predicted_Probability"]),
            "CI_Lower_95": float(pred["CI_Lower_95"]),
            "CI_Upper_95": float(pred["CI_Upper_95"]),
        })

    return pd.DataFrame(rows).sort_values("Predicted_Probability").reset_index(drop=True)


def run_sex_sensitivity(
    res,
    baseline: pd.DataFrame,
    df_training_features: pd.DataFrame,
    features: list[str],
    model_columns: list[str],
) -> pd.DataFrame:
    """
    Compare female and male sensitivity around the baseline profile.
    """
    female = baseline.copy()
    female["P_SEX"] = "F"
    male = baseline.copy()
    male["P_SEX"] = "M"

    scored = predict_profile_probabilities(
        res,
        pd.concat([female, male], ignore_index=True),
        df_training_features,
        features,
        model_columns,
    )

    female_prob = float(scored.iloc[0]["Predicted_Probability"])
    records = [
        {
            "Scenario": "Female baseline",
            "P_SEX": "F",
            "Predicted_Probability": female_prob,
            "CI_Lower_95": float(scored.iloc[0]["CI_Lower_95"]),
            "CI_Upper_95": float(scored.iloc[0]["CI_Upper_95"]),
            "Absolute_Change_vs_F": 0.0,
            "Relative_Change_Percent_vs_F": 0.0,
        },
        {
            "Scenario": "Male sensitivity",
            "P_SEX": "M",
            "Predicted_Probability": float(scored.iloc[1]["Predicted_Probability"]),
            "CI_Lower_95": float(scored.iloc[1]["CI_Lower_95"]),
            "CI_Upper_95": float(scored.iloc[1]["CI_Upper_95"]),
            "Absolute_Change_vs_F": float(scored.iloc[1]["Predicted_Probability"]) - female_prob,
            "Relative_Change_Percent_vs_F": ((float(scored.iloc[1]["Predicted_Probability"]) - female_prob) / female_prob * 100.0) if female_prob != 0 else np.nan,
        },
    ]
    return pd.DataFrame(records)


def build_combined_or_table(res) -> pd.DataFrame:
    """
    Format the combined-model odds ratios for downstream CSVs and plots.
    """
    or_table = make_or_table(res, model_name="Combined Binary Model", top_n=25).reset_index()
    or_table = or_table.rename(
        columns={
            "index": "Term",
            "OR": "Odds_Ratio",
            "CI_low": "CI_Lower_95",
            "CI_high": "CI_Upper_95",
        }
    )

    if "p_value" not in or_table.columns:
        or_table["p_value"] = np.nan

    or_table.insert(1, "Coefficient", or_table["Term"].map(res.params.to_dict()))
    return or_table[["Term", "Coefficient", "Odds_Ratio", "CI_Lower_95", "CI_Upper_95", "p_value"]]


def generate_simulation_outputs(*, save_outputs: bool = True) -> dict[str, pd.DataFrame]:
    """
    Generate the simulation tables consumed by visualizations.py.
    """
    res, df_c, features, categorical_cols, model_columns, fit_type = fit_combined_binary_model()
    baseline = make_baseline_profile(df_c, features)
    baseline_profile = score_profiles(res, baseline, df_c, features, model_columns)
    category_switch = run_category_switch_simulations(res, baseline, df_c, features, model_columns)
    sex_sensitivity = run_sex_sensitivity(res, baseline, df_c, features, model_columns)
    vehicle_age_sensitivity = run_continuous_sensitivity(
        res,
        baseline,
        df_c,
        features,
        model_columns,
        variable="VEHICLE_AGE",
        values=list(range(0, 21)),
    )
    person_age_sensitivity = run_continuous_sensitivity(
        res,
        baseline,
        df_c,
        features,
        model_columns,
        variable="P_AGE",
        values=list(range(16, 81, 4)),
    )
    risk_profiles = run_risk_profiles(res, baseline, df_c, features, model_columns)
    combined_or_table = build_combined_or_table(res)

    outputs = {
        "baseline_profile": baseline_profile,
        "category_switch": category_switch,
        "sex_sensitivity": sex_sensitivity,
        "vehicle_age_sensitivity": vehicle_age_sensitivity,
        "person_age_sensitivity": person_age_sensitivity,
        "risk_profiles": risk_profiles,
        "combined_or_table": combined_or_table,
    }

    if save_outputs:
        ensure_output_dir()
        baseline_profile.to_csv(OUTPUT_DIR / "baseline_profile.csv", index=False)
        category_switch.to_csv(OUTPUT_DIR / "category_switch_simulation.csv", index=False)
        sex_sensitivity.to_csv(OUTPUT_DIR / "sex_sensitivity.csv", index=False)
        vehicle_age_sensitivity.to_csv(OUTPUT_DIR / "vehicle_age_sensitivity.csv", index=False)
        person_age_sensitivity.to_csv(OUTPUT_DIR / "person_age_sensitivity.csv", index=False)
        risk_profiles.to_csv(OUTPUT_DIR / "risk_profiles.csv", index=False)
        combined_or_table.to_csv(OUTPUT_DIR / "combined_binary_odds_ratios.csv", index=False)

    return outputs


def main():
    print("\nFitting combined binary model for simulation...")
    res, df_c, features, categorical_cols, model_columns, fit_type = fit_combined_binary_model()

    baseline = make_baseline_profile(df_c, features)
    baseline_scored = score_profiles(res, baseline, df_c, features, model_columns)

    print(f"\nFit type: {fit_type}")
    print("\nBaseline profile with prediction interval:")
    print(baseline_scored.round(6))

    cat_results = run_category_switch_simulations(res, baseline, df_c, features, model_columns)
    print("\nScenario comparison:")
    print(cat_results.round(4))

    sex_results = run_sex_sensitivity(res, baseline, df_c, features, model_columns)
    print("\nSex sensitivity:")
    print(sex_results.round(4))

    veh_age_results = run_continuous_sensitivity(
        res, baseline, df_c, features, model_columns,
        variable="VEHICLE_AGE",
        values=list(range(0, 21))
    )
    print("\nVEHICLE_AGE sensitivity:")
    print(veh_age_results.head())

    p_age_results = run_continuous_sensitivity(
        res, baseline, df_c, features, model_columns,
        variable="P_AGE",
        values=list(range(16, 81, 4))
    )
    print("\nP_AGE sensitivity:")
    print(p_age_results.head().round(4))

    risk_profiles = run_risk_profiles(res, baseline, df_c, features, model_columns)
    print("\nRisk profile comparison:")
    print(risk_profiles.round(4))


if __name__ == "__main__":
    main()
