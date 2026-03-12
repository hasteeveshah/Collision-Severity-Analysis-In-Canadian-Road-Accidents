import os
import time

import pandas as pd

from Load_prepare import RANDOM_STATE
from models import preprocess_categoricals_variablewise
from utils import (
    dataframe_to_pretty_table,
    fit_eval_ordered_logit,
    make_or_table_ordered,
    model_comparison_pretty,
)


def run_ordinal_models(df_input: pd.DataFrame | None = None, fast: bool = False):
    df = pd.read_parquet("NCDB.parquet") if df_input is None else df_input.copy()
    t0 = time.perf_counter()
    print(f"Loaded data: {len(df):,} rows")
    sample_n = 75000 if df_input is None else None
    base_df = df[df["P_ISEV"].isin([1, 2, 3])].copy()

    if sample_n is not None and len(base_df) > sample_n:
        base_df = base_df.sample(sample_n, random_state=RANDOM_STATE).copy()
        print(f"Using sampled ordinal dataset: {len(base_df):,} rows")
    else:
        print(f"Using ordinal dataset: {len(base_df):,} rows")

    if fast:
        rq1_maxiter = 25
        rq2_maxiter = 30
        combined_maxiter = 35
        print(f"Running in FAST mode (ORDINAL_FAST=1), sample_n={sample_n if sample_n is not None else 'external'}")
    else:
        rq1_maxiter = 200
        rq2_maxiter = 250
        combined_maxiter = 300
        print(f"Running in FULL mode (ORDINAL_FAST=0), sample_n={sample_n if sample_n is not None else 'external'}")

    # RQ1 (Ordinal): Environment and Traffic Context Factors Affecting Crash Severity
    rq1_features = ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF", "C_MNTH", "C_WDAY", "C_HOUR", "C_VEHS"]
    rq1_numeric = ["C_MNTH", "C_WDAY", "C_HOUR", "C_VEHS"]
    rq1_cat = ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF"]

    # RQ2 (Ordinal): Human/Vehicle -> Injury Severity (P_ISEV 1/2/3)
    rq2_features = ["P_SAFE", "P_USER", "P_SEX", "V_TYPE", "P_AGE", "V_YEAR", "C_YEAR"]
    rq2_numeric = ["P_AGE", "V_YEAR", "C_YEAR"]
    rq2_cat = ["P_SAFE", "P_USER", "P_SEX", "V_TYPE"]

    # Combined (Ordinal): Env + Human/Vehicle -> P_ISEV
    combined_features = rq1_features + rq2_features
    combined_numeric = rq1_numeric + rq2_numeric
    combined_cat = rq1_cat + rq2_cat

    metrics_list = []

    def prepare_variant_df(
        df_local: pd.DataFrame,
        *,
        features: list[str],
        categorical_cols: list[str],
        include_other_group: bool,
    ) -> pd.DataFrame:
        df_variant = df_local.dropna(subset=features + ["P_ISEV"]).copy()
        return preprocess_categoricals_variablewise(
            df_variant,
            categorical_cols,
            include_other_group=include_other_group,
            default_rare_threshold=3000,
            default_keep_levels={"OTHER"},
        )

    def run_ordinal_variant(
        *,
        variant_label: str,
        include_other_group: bool,
    ) -> None:
        print(f"\n================ Ordinal Models ({variant_label}) ================")

        print("\nRQ1 ordinal model...", flush=True)
        t1 = time.perf_counter()
        df_rq1 = prepare_variant_df(
            base_df,
            features=rq1_features,
            categorical_cols=rq1_cat,
            include_other_group=include_other_group,
        )
        res_rq1, m_rq1 = fit_eval_ordered_logit(
            df=df_rq1,
            features=rq1_features,
            y_col="P_ISEV",
            title=f"RQ1 ordinal model ({variant_label})",
            sample_n=None,
            min_count_rare=3000,
            rare_cols=[],
            numeric_cols=rq1_numeric,
            keep_as_is_cols=rq1_cat,
            maxiter=rq1_maxiter,
            method="lbfgs",
        )
        print(f"RQ1 completed in {time.perf_counter() - t1:.1f}s", flush=True)
        _ = make_or_table_ordered(res_rq1, model_name=f"RQ1 ordinal model ({variant_label})")
        metrics_list.append(m_rq1)

        print("\nRQ2 ordinal model...", flush=True)
        t2 = time.perf_counter()
        df_rq2 = prepare_variant_df(
            base_df,
            features=rq2_features,
            categorical_cols=rq2_cat,
            include_other_group=include_other_group,
        )
        res_rq2, m_rq2 = fit_eval_ordered_logit(
            df=df_rq2,
            features=rq2_features,
            y_col="P_ISEV",
            title=f"RQ2 ordinal model ({variant_label})",
            sample_n=None,
            min_count_rare=3000,
            rare_cols=[],
            numeric_cols=rq2_numeric,
            keep_as_is_cols=rq2_cat,
            maxiter=rq2_maxiter,
            method="lbfgs",
        )
        print(f"RQ2 completed in {time.perf_counter() - t2:.1f}s", flush=True)
        _ = make_or_table_ordered(res_rq2, model_name=f"RQ2 ordinal model ({variant_label})")
        metrics_list.append(m_rq2)

        print("\nCombined ordinal model...", flush=True)
        t3 = time.perf_counter()
        df_c = prepare_variant_df(
            base_df,
            features=combined_features,
            categorical_cols=combined_cat,
            include_other_group=include_other_group,
        )
        res_c, m_c = fit_eval_ordered_logit(
            df=df_c,
            features=combined_features,
            y_col="P_ISEV",
            title=f"Combined ordinal model ({variant_label})",
            sample_n=None,
            min_count_rare=3000,
            rare_cols=[],
            numeric_cols=combined_numeric,
            keep_as_is_cols=combined_cat,
            maxiter=combined_maxiter,
            method="lbfgs",
        )
        print(f"Combined completed in {time.perf_counter() - t3:.1f}s", flush=True)
        _ = make_or_table_ordered(res_c, model_name=f"Combined ordinal model ({variant_label})")
        metrics_list.append(m_c)

    run_ordinal_variant(variant_label="with_other_group", include_other_group=True)
    run_ordinal_variant(variant_label="without_other_group", include_other_group=False)

    comparison = model_comparison_pretty(metrics_list)
    print("\n================ ORDINAL MODEL COMPARISON ================")
    print(dataframe_to_pretty_table(comparison.round(4)))
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")

    return comparison


if __name__ == "__main__":
    # FULL mode is the default. Set ORDINAL_FAST=1 to opt into the lighter run.
    fast_mode = os.getenv("ORDINAL_FAST", "0") == "1"
    run_ordinal_models(fast=fast_mode)
