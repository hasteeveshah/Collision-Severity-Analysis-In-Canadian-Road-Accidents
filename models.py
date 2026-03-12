import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Load_prepare import df, RANDOM_STATE, MIN_COUNT_RARE
from utils import (
    build_design_matrix,
    collapse_special_levels,
    comparison_table,
    dataframe_to_pretty_table,
    evaluate_binary_classifier,
    group_rare_levels,
    make_or_table,
    print_level_diagnostics,
    reduce_to_full_rank,
)

PERSON_MODEL_SAMPLE_N = 150_000
RQ1_TITLE = "Environment and Traffic Context Factors Affecting Crash Severity"

CATEGORY_KEEP_LEVELS = {
    "C_WTHR": {"1", "2", "3", "4", "5", "6", "7", "OTHER"},
    "C_RSUR": {"1", "2", "3", "4", "5", "6", "7", "OTHER"},
    "C_RALN": {"1", "2", "3", "4", "5", "6", "OTHER"},
    "C_TRAF": {"01", "02", "03", "04", "06", "08", "18", "OTHER"},
    "P_SAFE": {"01", "02", "09", "12", "13", "OTHER"},
    "P_USER": {"1", "2", "3", "4", "5", "OTHER"},
    "P_SEX": {"F", "M", "OTHER"},
    "V_TYPE": {"01", "05", "06", "07", "08", "09", "11", "14", "17", "OTHER"},
}

CATEGORY_RARE_THRESHOLDS = {
    "C_WTHR": 100,
    "C_RSUR": 100,
    "C_RALN": 100,
    "C_TRAF": 150,
    "P_SAFE": 100,
    "P_USER": 75,
    "P_SEX": 25,
    "V_TYPE": 125,
}


def preprocess_categoricals_variablewise(
    df_local: pd.DataFrame,
    categorical_cols: list[str],
    *,
    include_other_group: bool,
    default_rare_threshold: int | None = None,
    default_keep_levels: set[str] | None = None,
) -> pd.DataFrame:
    fallback_threshold = MIN_COUNT_RARE if default_rare_threshold is None else default_rare_threshold
    fallback_keep_levels = {"OTHER"} if default_keep_levels is None else set(default_keep_levels)

    df_local = df_local.copy()
    for col in categorical_cols:
        keep_levels = set(CATEGORY_KEEP_LEVELS.get(col, fallback_keep_levels))
        rare_threshold = int(CATEGORY_RARE_THRESHOLDS.get(col, fallback_threshold))

        df_local = collapse_special_levels(df_local, col, other_label="OTHER")
        df_local = group_rare_levels(
            df_local,
            col,
            min_count=rare_threshold,
            other_label="OTHER",
            keep_levels=keep_levels,
        )

    if not include_other_group and categorical_cols:
        keep_mask = pd.Series(True, index=df_local.index)
        for col in categorical_cols:
            keep_mask &= df_local[col].astype("string").ne("OTHER")
        df_local = df_local.loc[keep_mask].copy()

    return df_local


def impute_numeric_with_missing_flags(
    df_local: pd.DataFrame,
    numeric_cols: list[str],
    *,
    indicator_suffix: str = "_MISSING",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Preserve rows with structurally missing numeric features by imputing the
    numeric value and tracking the missingness explicitly.
    """
    df_local = df_local.copy()
    added_cols: list[str] = []

    for col in numeric_cols:
        series = pd.to_numeric(df_local[col], errors="coerce")
        missing_mask = series.isna()

        if missing_mask.any():
            indicator_col = f"{col}{indicator_suffix}"
            df_local[indicator_col] = missing_mask.astype("int8")
            added_cols.append(indicator_col)

        fill_value = float(series.median()) if series.notna().any() else 0.0
        df_local[col] = series.fillna(fill_value).astype("float32")

    return df_local, added_cols


def run_person_level_models(
    df_input: pd.DataFrame | None = None,
    sample_n: int | None = PERSON_MODEL_SAMPLE_N,
):
    results_for_table = []

    # Person-level filter
    source_df = df if df_input is None else df_input
    df_person = source_df[source_df["P_ISEV"].isin([1, 2, 3])].copy()
    df_person["Y_INJURY"] = df_person["P_ISEV"].isin([2, 3]).astype(int)

    if sample_n is not None and len(df_person) > sample_n:
        df_person = df_person.sample(sample_n, random_state=RANDOM_STATE).copy()
        print(f"Using sampled person-level dataset: {len(df_person):,} rows")
    else:
        print(f"Using person-level dataset: {len(df_person):,} rows")

    def preprocess_categoricals(
        df_local: pd.DataFrame,
        categorical_cols: list[str],
        *,
        include_other_group: bool,
        rare_threshold: int | None = None,
        keep_levels: set[str] | None = None,
    ) -> pd.DataFrame:
        return preprocess_categoricals_variablewise(
            df_local,
            categorical_cols,
            include_other_group=include_other_group,
            default_rare_threshold=rare_threshold,
            default_keep_levels=keep_levels,
        )

    def fit_binary_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        model_label: str,
        maxiter: int = 300,
        allow_regularized: bool = True,
    ):
        model = sm.Logit(y_train, X_train)
        fit_type = "MLE"
        solver_name = "newton"
        res = None
        last_mle_result = None
        last_mle_solver = None
        last_mle_error = None

        print(
            f"Fitting {model_label}: "
            f"{X_train.shape[0]:,} train rows x {X_train.shape[1]} features "
            f"({X_test.shape[0]:,} test rows)"
        )

        mle_attempts = [
            ("lbfgs", max(maxiter, 800)),
            ("bfgs", max(maxiter, 800)),
            ("newton", maxiter),
            ("cg", max(maxiter, 800)),
        ]
        for method_name, iter_budget in mle_attempts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", HessianInversionWarning)
                    candidate = model.fit(method=method_name, maxiter=iter_budget, disp=0)
                solver_name = method_name
                mle_retvals = getattr(candidate, "mle_retvals", {}) or {}
                converged = bool(mle_retvals.get("converged", getattr(candidate, "converged", True)))
                if converged:
                    res = candidate
                    break

                print(f"\n{model_label} MLE with {method_name} did not fully converge. Trying another MLE solver.")
                last_mle_result = candidate
                last_mle_solver = method_name
            except Exception as mle_error:
                print(f"\n{model_label} MLE with {method_name} failed.")
                print("Error:", repr(mle_error))
                last_mle_error = mle_error

        if res is None and last_mle_result is not None and not allow_regularized:
            res = last_mle_result
            solver_name = last_mle_solver or solver_name
            print(f"\n{model_label} is using the best available MLE result from {solver_name}.")

        if res is None:
            if allow_regularized:
                print(f"\n{model_label} MLE solvers failed. Switching to regularized sensitivity fit.")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", HessianInversionWarning)
                        res = model.fit_regularized(method="l1", alpha=0.01, maxiter=5000, disp=0)
                    fit_type = "Regularized (L1)"
                    solver_name = "l1"
                except Exception as regularized_error:
                    print(f"\n{model_label} regularized statsmodels fit also failed. Switching to sklearn L2 fit.")
                    print("Error:", repr(regularized_error))
                    sk_model = LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    )
                    sk_model.fit(X_train, y_train)
                    res = sk_model
                    fit_type = "Sklearn (L2)"
                    solver_name = "liblinear"
            else:
                raise RuntimeError(
                    f"{model_label} could not be fit with any MLE solver."
                ) from last_mle_error

        print(f"\n{model_label} fitted using: {fit_type} [{solver_name}]")
        if fit_type == "Sklearn (L2)":
            p = res.predict_proba(X_test)[:, 1]
        else:
            p = res.predict(X_test)
        metrics = evaluate_binary_classifier(y_test, p, threshold=0.50, title=f"{model_label} ({fit_type})")

        if fit_type == "MLE":
            make_or_table(res, model_name=f"{model_label} (MLE)", top_n=25)
        elif fit_type == "Sklearn (L2)":
            or_table = pd.DataFrame(
                {"OR": np.exp(pd.Series(res.coef_.ravel(), index=X_train.columns))}
            ).sort_values("OR", ascending=False)
            print(f"\n================ {model_label} Odds Ratios (Sklearn L2) =================")
            print(or_table.head(25).round(4))
            print("\nBottom 25 (most protective):")
            print(or_table.tail(25).round(4))
        else:
            or_table = pd.DataFrame({"OR": np.exp(res.params)}).sort_values("OR", ascending=False)
            print(f"\n================ {model_label} Odds Ratios (Regularized) =================")
            print(or_table.head(25).round(4))
            print("\nBottom 25 (most protective):")
            print(or_table.tail(25).round(4))

        return res, metrics, fit_type

    def split_and_clean_design(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        stabilize_sparse_dummies: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        def is_binary_dummy(series: pd.Series) -> bool:
            non_na = pd.Series(series).dropna()
            if non_na.empty:
                return False
            unique_vals = pd.Index(non_na.unique())
            return bool(unique_vals.isin([0, 1, 0.0, 1.0, False, True]).all())

        def drop_sparse_or_separating_dummies(
            X_local: pd.DataFrame,
            y_local: pd.Series,
        ) -> tuple[pd.DataFrame, list[str]]:
            min_active_support = max(8, int(0.0005 * len(X_local)))
            min_active_class_support = max(2, int(0.00005 * len(X_local)))

            keep_cols: list[str] = []
            for col in X_local.columns:
                if col == "const":
                    keep_cols.append(col)
                    continue

                col_series = X_local[col]
                if not is_binary_dummy(col_series):
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

            return X_local[keep_cols].copy(), keep_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
        )

        X_train = X_train.loc[:, X_train.nunique(dropna=False) > 1]
        X_train = X_train.loc[:, ~X_train.T.duplicated()]
        if stabilize_sparse_dummies:
            X_train, _ = drop_sparse_or_separating_dummies(X_train, y_train)
            X_train = X_train.loc[:, X_train.nunique(dropna=False) > 1]
            X_train = X_train.loc[:, ~X_train.T.duplicated()]
        X_train, keep_cols = reduce_to_full_rank(X_train)
        X_test = X_test.reindex(columns=keep_cols, fill_value=0.0)

        return X_train, X_test, y_train, y_test

    rq1_features = ["C_WTHR","C_RSUR","C_RALN","C_TRAF","C_MNTH","C_WDAY","C_HOUR","C_VEHS"]
    rq1_cat = ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF"]
    rq2_features = ["P_SAFE", "P_USER", "P_SEX", "V_TYPE", "VEHICLE_AGE", "P_AGE"]
    rq2_cat = ["P_SAFE", "P_USER", "P_SEX", "V_TYPE"]
    rq2_numeric = ["VEHICLE_AGE", "P_AGE"]
    combined_features = rq1_features + rq2_features
    combined_cat = ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF", "P_SAFE", "P_USER", "P_SEX", "V_TYPE"]
    combined_impute_numeric = ["VEHICLE_AGE", "P_AGE"]

    def run_model_variant(
        *,
        variant_label: str,
        include_other_group: bool,
    ) -> None:
        print(f"\n================ Binary Models ({variant_label}) =================")

        # ---------------- RQ1: Environment and Traffic Context ----------------
        print(f"\n================ RQ1: {RQ1_TITLE} =================")
        df_rq1 = df_person.dropna(subset=rq1_features + ["Y_INJURY"]).copy()
        df_rq1 = preprocess_categoricals(
            df_rq1,
            rq1_cat,
            include_other_group=include_other_group,
            rare_threshold=max(MIN_COUNT_RARE, 5000),
            keep_levels={"OTHER"},
        )

        for col in rq1_cat:
            print_level_diagnostics(df_rq1, col, "Y_INJURY", top_n=8)

        y1 = df_rq1["Y_INJURY"].astype(int)
        X1 = build_design_matrix(df_rq1, rq1_features, drop_first=True, add_constant=True)
        X1, _ = reduce_to_full_rank(X1)

        X1_train, X1_test, y1_train, y1_test = split_and_clean_design(
            X1,
            y1,
            stabilize_sparse_dummies=True,
        )
        print("RQ1 shapes:", X1_train.shape, X1_test.shape)
        _, r1, _ = fit_binary_model(
            X1_train,
            y1_train,
            X1_test,
            y1_test,
            model_label=f"RQ1 {RQ1_TITLE} ({variant_label})",
            allow_regularized=False,
        )
        r1["Model"] = f"RQ1 {RQ1_TITLE} ({variant_label})"
        results_for_table.append(r1)

        # ---------------- RQ2: Human & Vehicle Model ----------------
        print("\n================ RQ2: Human & Vehicle Model =================")
        df_rq2 = df_person.dropna(subset=rq2_cat + ["Y_INJURY"]).copy()
        df_rq2, rq2_missing_flags = impute_numeric_with_missing_flags(df_rq2, rq2_numeric)
        df_rq2 = preprocess_categoricals(
            df_rq2,
            rq2_cat,
            include_other_group=include_other_group,
            rare_threshold=20000,
            keep_levels={"OTHER"},
        )
        rq2_model_features = rq2_features + rq2_missing_flags

        for col in rq2_cat:
            print_level_diagnostics(df_rq2, col, "Y_INJURY", top_n=8)

        y2 = df_rq2["Y_INJURY"].astype(int)
        X2 = build_design_matrix(df_rq2, rq2_model_features, drop_first=True, add_constant=True)
        X2, _ = reduce_to_full_rank(X2)

        X2_train, X2_test, y2_train, y2_test = split_and_clean_design(
            X2,
            y2,
            stabilize_sparse_dummies=True,
        )

        print("RQ2 shapes:", X2_train.shape, X2_test.shape)
        res2, r2, rq2_fit_type = fit_binary_model(
            X2_train,
            y2_train,
            X2_test,
            y2_test,
            model_label=f"RQ2 Human/Vehicle ({variant_label})",
            allow_regularized=False,
        )
        if rq2_fit_type == "Regularized (L1)":
            print("Non-zero coefficients:", int((res2.params != 0).sum()), "out of", len(res2.params))
        r2["Model"] = f"RQ2 Human/Vehicle ({variant_label})"
        results_for_table.append(r2)

        # ---------------- Combined Model ----------------
        print("\n================ Combined Model =================")
        combined_required = [col for col in combined_features if col not in set(combined_impute_numeric)]
        df_c = df_person.dropna(subset=combined_required + ["Y_INJURY"]).copy()
        df_c, combined_missing_flags = impute_numeric_with_missing_flags(df_c, combined_impute_numeric)
        df_c = preprocess_categoricals(
            df_c,
            combined_cat,
            include_other_group=include_other_group,
            rare_threshold=max(MIN_COUNT_RARE, 5000),
            keep_levels={"OTHER"},
        )
        combined_model_features = combined_features + combined_missing_flags

        y3 = df_c["Y_INJURY"].astype(int)
        X3 = build_design_matrix(df_c, combined_model_features, drop_first=True, add_constant=True)

        # cleanup: zero-variance + duplicate columns
        X3 = X3.loc[:, X3.nunique() > 1]
        X3 = X3.loc[:, ~X3.T.duplicated()]
        X3, _ = reduce_to_full_rank(X3)

        X3_train, X3_test, y3_train, y3_test = split_and_clean_design(
            X3,
            y3,
            stabilize_sparse_dummies=True,
        )

        print("Combined shapes:", X3_train.shape, X3_test.shape)
        _, r3, _ = fit_binary_model(
            X3_train,
            y3_train,
            X3_test,
            y3_test,
            model_label=f"Combined ({variant_label})",
            allow_regularized=False,
        )
        r3["Model"] = f"Combined ({variant_label})"
        results_for_table.append(r3)

    run_model_variant(variant_label="with_other_group", include_other_group=True)
    run_model_variant(variant_label="without_other_group", include_other_group=False)

    # ---------------- Comparison Table ----------------
    print("\n================ Comparison Table =================")
    comp = comparison_table(results_for_table)
    print(dataframe_to_pretty_table(comp.round(4)))

    return comp

if __name__ == "__main__":
    run_person_level_models()
