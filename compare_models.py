from pathlib import Path

import pandas as pd

from Load_prepare import N_SAMPLE_MAIN, RANDOM_STATE, df
from models import run_person_level_models
from ordinal_regression import run_ordinal_models
from utils import dataframe_to_pretty_table

OUTPUT_DIR = Path("outputs/model_comparison")


def standardize_binary(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize binary model output columns so they merge cleanly.
    """
    df_out = df_in.copy()
    rename_map = {
        "model": "Model",
        "Model_Name": "Model",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced_Accuracy",
        "macro_f1": "Macro_F1",
        "roc_auc": "ROC_AUC",
        "mae": "MAE",
    }
    df_out = df_out.rename(columns=rename_map)

    required_cols = ["Model", "Accuracy", "Balanced_Accuracy", "Macro_F1", "ROC_AUC", "MAE"]
    for col in required_cols:
        if col not in df_out.columns:
            df_out[col] = pd.NA

    df_out["Family"] = "Binary Logit"
    return df_out[["Model", "Family", "Accuracy", "Balanced_Accuracy", "Macro_F1", "ROC_AUC", "MAE"]]


def standardize_ordinal(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize ordinal model output columns so they match the binary output.
    """
    df_out = df_in.copy()
    rename_map = {
        "model": "Model",
        "Model_Name": "Model",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced_Accuracy",
        "macro_f1": "Macro_F1",
        "roc_auc": "ROC_AUC",
        "mae": "MAE",
    }
    df_out = df_out.rename(columns=rename_map)

    required_cols = ["Model", "Accuracy", "Balanced_Accuracy", "Macro_F1", "ROC_AUC", "MAE"]
    for col in required_cols:
        if col not in df_out.columns:
            df_out[col] = pd.NA

    df_out["Family"] = "Ordinal Logit"
    return df_out[["Model", "Family", "Accuracy", "Balanced_Accuracy", "Macro_F1", "ROC_AUC", "MAE"]]


def build_model_comparison(save_output: bool = True) -> pd.DataFrame:
    """
    Fit binary and ordinal models on the same sampled data and return one table.
    """
    base_df = df[df["P_ISEV"].isin([1, 2, 3])].copy()
    if len(base_df) > N_SAMPLE_MAIN:
        shared_sample = base_df.sample(N_SAMPLE_MAIN, random_state=RANDOM_STATE).copy()
    else:
        shared_sample = base_df.copy()

    print(f"\nUsing shared sampled dataset for comparison: {len(shared_sample):,} rows")

    print("\nRunning binary logistic models...")
    binary_results = standardize_binary(run_person_level_models(df_input=shared_sample))

    print("\nRunning ordinal logistic models...")
    ordinal_results = standardize_ordinal(run_ordinal_models(df_input=shared_sample))

    final_df = pd.concat([binary_results, ordinal_results], ignore_index=True)

    if final_df["ROC_AUC"].notna().any():
        final_df = final_df.sort_values(by="ROC_AUC", ascending=False, na_position="last")
    else:
        final_df = final_df.sort_values(by="Accuracy", ascending=False, na_position="last")

    final_df = final_df.reset_index(drop=True)

    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    return final_df


def main() -> None:
    final_df = build_model_comparison(save_output=True)
    print("\nFinal combined model comparison:\n")
    print(dataframe_to_pretty_table(final_df.round(4)))


if __name__ == "__main__":
    main()
