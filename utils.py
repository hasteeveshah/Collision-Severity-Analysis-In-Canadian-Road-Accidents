from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from prettytable import PrettyTable
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
)

# Block A — Categorical hygiene

DEFAULT_KEEP_LEVELS = {"U", "UU", "Q", "QQ", "N", "NN", "R", "RR", "X", "XX"}
DEFAULT_SPECIAL_LEVELS = DEFAULT_KEEP_LEVELS | {"RARE"}


def dataframe_to_pretty_table(
    df_in: pd.DataFrame,
    *,
    float_precision: int = 4,
    max_widths: dict[str, int] | None = None,
) -> str:
    """
    Render a DataFrame as a PrettyTable string for console display only.
    """
    table = PrettyTable()
    table.field_names = [str(col) for col in df_in.columns]
    table.align = "l"

    default_widths = {"Model": 48, "Term": 42, "Confusion_Matrix": 28}
    for col, width in {**default_widths, **(max_widths or {})}.items():
        if col in table.field_names:
            table.max_width[col] = width

    for row in df_in.itertuples(index=False, name=None):
        formatted_row: list[str] = []
        for value in row:
            if value is None:
                formatted_row.append("")
            elif isinstance(value, (list, tuple, dict, np.ndarray)):
                formatted_row.append(str(value))
            elif isinstance(value, (float, np.floating)):
                formatted_row.append(f"{float(value):.{float_precision}f}")
            elif pd.isna(value):
                formatted_row.append("")
            else:
                formatted_row.append(str(value))
        table.add_row(formatted_row)

    return table.get_string()


def confusion_matrix_to_pretty_table(
    matrix: list[list[int]] | np.ndarray,
    *,
    labels: list[object] | np.ndarray | None = None,
) -> str:
    """
    Render a confusion matrix as a boxed table for console display.
    """
    matrix_arr = np.asarray(matrix)
    if matrix_arr.ndim != 2:
        return str(matrix_arr)

    n_rows, n_cols = matrix_arr.shape
    if labels is None:
        labels = list(range(max(n_rows, n_cols)))
    else:
        labels = [str(label) for label in labels]

    if len(labels) < n_cols:
        labels = labels + [str(idx) for idx in range(len(labels), n_cols)]

    row_labels = labels[:n_rows]
    col_labels = labels[:n_cols]
    pred_headers: list[str] = []
    seen_headers: set[str] = set()
    for idx, label in enumerate(col_labels):
        header = f"Pred {label}"
        if header in seen_headers:
            header = f"Pred {idx} ({label})"
        seen_headers.add(header)
        pred_headers.append(header)

    display_df = pd.DataFrame(matrix_arr, columns=pred_headers)
    display_df.insert(0, "Actual", [f"Actual {label}" for label in row_labels])
    return dataframe_to_pretty_table(display_df, float_precision=0, max_widths={"Actual": 14})


def collapse_special_levels(
    df: pd.DataFrame,
    col: str,
    *,
    levels: set[str] | None = None,
    other_label: str = "OTHER",
) -> pd.DataFrame:
    """
    Collapse unknown/special codes into one interpretable bucket.
    """
    collapse_levels = DEFAULT_SPECIAL_LEVELS if levels is None else set(levels)
    s = df[col].astype("string")
    df[col] = s.where(~s.isin(list(collapse_levels)), other=other_label)
    return df


def print_level_diagnostics(
    df: pd.DataFrame,
    col: str,
    y_col: str,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Print simple outcome-rate diagnostics by category level.
    """
    summary = df.groupby(col, dropna=False)[y_col].agg(["count", "sum"])
    summary["event_rate"] = summary["sum"] / summary["count"]
    summary = summary[["count", "event_rate"]].sort_values(["event_rate", "count"], ascending=[False, False])

    print(f"\n{col} outcome-rate diagnostics:")
    display_df = summary.head(top_n).round(4).reset_index()
    print(dataframe_to_pretty_table(display_df))

    return summary


def group_rare_levels(
    df: pd.DataFrame,
    col: str,
    min_count: int = 3000,
    other_label: str = "RARE",
    keep_levels: set[str] | None = None,
) -> pd.DataFrame:

    keep = DEFAULT_KEEP_LEVELS if keep_levels is None else set(keep_levels)

    s = df[col].astype("string")
    vc = s.value_counts(dropna=False)

    # Rare = under min_count and not in keep list
    rare_levels = set(vc[vc < min_count].index.astype("string"))
    rare_levels = {lvl for lvl in rare_levels if lvl not in keep and lvl != "<NA>"}

    df[col] = s.where(~s.isin(list(rare_levels)), other=other_label)
    return df


# Block B — Design matrices

def build_design_matrix(
    df: pd.DataFrame,
    features: list[str],
    drop_first: bool = True,
    add_constant: bool = True,
    dtype: str = "float32",
) -> pd.DataFrame:

    X = pd.get_dummies(df[features], drop_first=drop_first)
    X = X.apply(pd.to_numeric, errors="coerce").astype(dtype)

    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    # Remove zero-variance & duplicate columns (safety)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.loc[:, ~X.T.duplicated()]

    return X


def build_design_matrix_for_ordinal(
    df: pd.DataFrame,
    features: list[str],
    drop_first: bool = True,
    dtype: str = "float32",
) -> pd.DataFrame:

    X = pd.get_dummies(df[features], drop_first=drop_first)
    X = X.apply(pd.to_numeric, errors="coerce").astype(dtype)

    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.loc[:, ~X.T.duplicated()]

    return X


def reduce_to_full_rank(
    X: pd.DataFrame,
    *,
    tol: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop linearly dependent columns while preserving column order.
    """
    if X.empty:
        return X.copy(), []

    keep_cols: list[str] = []
    current_rank = 0

    for col in X.columns:
        trial_cols = keep_cols + [col]
        trial_rank = np.linalg.matrix_rank(X[trial_cols].to_numpy(dtype="float64"), tol=tol)
        if trial_rank > current_rank:
            keep_cols.append(col)
            current_rank = trial_rank

    return X[keep_cols].copy(), keep_cols

# Block C — Odds-ratio tables


def make_or_table(res, model_name: str = "Model", top_n: int = 25) -> pd.DataFrame:
    """
    Odds ratio table for statsmodels Logit (MLE fits).
    """
    params = res.params

    try:
        conf = res.conf_int()
        conf.columns = ["CI_low", "CI_high"]
        pvals = res.pvalues

        or_table = pd.DataFrame(
            {
                "OR": np.exp(params),
                "CI_low": np.exp(conf["CI_low"]),
                "CI_high": np.exp(conf["CI_high"]),
                "p_value": pvals,
            }
        ).sort_values("OR", ascending=False)

    except Exception as e:
        print(f"\nNote: CI/p-values not available for {model_name} ({repr(e)}). Showing OR only.")
        or_table = pd.DataFrame({"OR": np.exp(params)}).sort_values("OR", ascending=False)

    print(f"\n================ {model_name} Odds Ratios =================")
    top_display = or_table.head(top_n).round(4).reset_index().rename(columns={"index": "Term"})
    print(dataframe_to_pretty_table(top_display))
    print("\nBottom 15 (most protective):")
    bottom_display = or_table.tail(15).round(4).reset_index().rename(columns={"index": "Term"})
    print(dataframe_to_pretty_table(bottom_display))

    return or_table


def make_or_table_ordinal(res, model_name: str = "Ordered Logit", top_n: int = 25) -> pd.DataFrame:

    params = res.params.copy()

    # For ordinal models, we typically focus on the "feature" parameters
    feature_params = params[~params.index.astype(str).str.contains(r"/", regex=True)]

    or_table = pd.DataFrame({"OR": np.exp(feature_params)}).sort_values("OR", ascending=False)

    print(f"\n================ {model_name} Odds Ratios (Ordinal) =================")
    top_display = or_table.head(top_n).round(4).reset_index().rename(columns={"index": "Term"})
    print(dataframe_to_pretty_table(top_display))
    print("\nBottom 15 (most protective):")
    bottom_display = or_table.tail(15).round(4).reset_index().rename(columns={"index": "Term"})
    print(dataframe_to_pretty_table(bottom_display))

    return or_table


# Alias for script's naming
make_or_table_ordered = make_or_table_ordinal


# Block D — Evaluation helpers

def evaluate_binary_classifier(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    title: str = "Model",
) -> dict:

    y_pred = (y_prob >= threshold).astype(int)

    # classification_report as dict gives stable scalar summaries
    rep_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    macro_prec = rep_dict.get("macro avg", {}).get("precision", np.nan)
    macro_rec = rep_dict.get("macro avg", {}).get("recall", np.nan)
    macro_f1 = rep_dict.get("macro avg", {}).get("f1-score", np.nan)

    out = {
        "Model": title,
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_Precision": float(macro_prec) if macro_prec == macro_prec else np.nan,
        "Macro_Recall": float(macro_rec) if macro_rec == macro_rec else np.nan,
        "Macro_F1": float(macro_f1) if macro_f1 == macro_f1 else np.nan,
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist(),
        # keep full text report separately (string)
        "Report_Text": classification_report(y_true, y_pred, zero_division=0),
    }
    cm_labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()])).tolist()

    print(f"\n=== Evaluation: {title} (threshold={threshold:.3f}) ===")
    print("Accuracy:", round(out["Accuracy"], 4))
    print("Balanced Accuracy:", round(out["Balanced_Accuracy"], 4))
    print("Macro Precision/Recall/F1:", round(out["Macro_Precision"], 4), round(out["Macro_Recall"], 4), round(out["Macro_F1"], 4))
    print("ROC AUC:", round(out["ROC_AUC"], 4))
    print("Confusion Matrix:")
    print(confusion_matrix_to_pretty_table(out["Confusion_Matrix"], labels=cm_labels))

    return out


def tune_threshold_for_balanced_accuracy(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    best_t, best_bacc = float(thresholds[0]), -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        bacc = balanced_accuracy_score(y_true, y_pred)
        if bacc > best_bacc:
            best_bacc, best_t = float(bacc), float(t)
    return best_t, best_bacc


def comparison_table(results: list[dict], sort_by: str = "ROC_AUC") -> pd.DataFrame:
   
    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results).copy()

    # Ensure strings stay strings (avoid NaN confusion)
    if "Report_Text" in df_out.columns:
        df_out["Report_Text"] = df_out["Report_Text"].fillna("").astype(str)

    # Order columns
    keep_cols = [
        "Model",
        "Threshold",
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "ROC_AUC",
        "MAE",
        "Confusion_Matrix",
    ]
    keep_cols = [c for c in keep_cols if c in df_out.columns]
    df_out = df_out[keep_cols]

    if sort_by in df_out.columns:
        df_out = df_out.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return df_out


# Alias forscript's naming
model_comparison_pretty = comparison_table

# Block E — Ordinal model helper (fixes your n_samples=0 issue)


def _fit_ordered_model_cleanly(
    model: OrderedModel,
    *,
    method: str,
    maxiter: int,
    disp: bool = False,
) -> tuple[object, bool, int]:
    """
    Fit an ordinal model while suppressing raw convergence warnings.
    Retry with larger iteration budgets before returning.
    """
    attempt_iters: list[int] = []
    current_iter = max(25, int(maxiter))
    for _ in range(3):
        if current_iter not in attempt_iters:
            attempt_iters.append(current_iter)
        current_iter *= 2

    last_res = None
    converged = False

    for iter_budget in attempt_iters:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", HessianInversionWarning)
            last_res = model.fit(method=method, maxiter=iter_budget, disp=disp)

        mle_retvals = getattr(last_res, "mle_retvals", {}) or {}
        converged = bool(mle_retvals.get("converged", getattr(last_res, "converged", False)))
        if converged:
            return last_res, True, iter_budget

    return last_res, converged, attempt_iters[-1]


def fit_and_evaluate_ordinal(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    link: str = "logit",
    method: str = "bfgs",
    maxiter: int = 200,
    title: str = "Ordered Logit",
) -> tuple[object, dict]:

    m = OrderedModel(y_train, X_train, distr=link)
    res, converged, used_maxiter = _fit_ordered_model_cleanly(
        m,
        method=method,
        maxiter=maxiter,
        disp=False,
    )
    if not converged:
        print(f"{title}: optimizer did not fully converge after maxiter={used_maxiter}.")

    # predicted class probabilities -> class with max prob
    p = res.model.predict(res.params, exog=X_test)
    class_labels = np.sort(pd.Series(y_train).dropna().astype(int).unique())
    if isinstance(p, pd.DataFrame):
        p_df = p.copy()
        p_df.columns = p_df.columns.astype(int)
        p_df = p_df.reindex(columns=class_labels, fill_value=0.0)
        y_pred = p_df.idxmax(axis=1).astype(int)
        y_score = p_df.to_numpy()
    else:
        p_arr = np.asarray(p)
        pred_idx = p_arr.argmax(axis=1)
        if len(class_labels) == p_arr.shape[1]:
            y_pred = pd.Series(class_labels[pred_idx], index=y_test.index).astype(int)
            y_score = p_arr
        else:
            # Fallback when labels cannot be inferred reliably
            y_pred = pd.Series(pred_idx, index=y_test.index).astype(int)
            y_score = p_arr

    y_test_arr = pd.Series(y_test).astype(int).to_numpy()
    y_pred_arr = pd.Series(y_pred).astype(int).to_numpy()

    rep_dict = classification_report(y_test_arr, y_pred_arr, zero_division=0, output_dict=True)
    macro_prec = rep_dict.get("macro avg", {}).get("precision", np.nan)
    macro_rec = rep_dict.get("macro avg", {}).get("recall", np.nan)
    macro_f1 = rep_dict.get("macro avg", {}).get("f1-score", np.nan)

    roc_auc = np.nan
    try:
        if len(class_labels) == 2:
            pos_idx = 1 if y_score.shape[1] > 1 else 0
            roc_auc = float(roc_auc_score(y_test_arr, y_score[:, pos_idx]))
        elif len(class_labels) > 2 and y_score.shape[1] == len(class_labels):
            y_test_bin = label_binarize(y_test_arr, classes=class_labels)
            roc_auc = float(roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = np.nan

    metrics = {
        "Model": title,
        "Accuracy": float(accuracy_score(y_test_arr, y_pred_arr)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test_arr, y_pred_arr)),
        "Macro_Precision": float(macro_prec) if macro_prec == macro_prec else np.nan,
        "Macro_Recall": float(macro_rec) if macro_rec == macro_rec else np.nan,
        "Macro_F1": float(macro_f1) if macro_f1 == macro_f1 else np.nan,
        "ROC_AUC": roc_auc,
        "MAE": float(mean_absolute_error(y_test_arr, y_pred_arr)),
        "Confusion_Matrix": confusion_matrix(y_test_arr, y_pred_arr).tolist(),
    }

    print(f"\n=== Ordinal Evaluation: {title} ===")
    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("Balanced Accuracy:", round(metrics["Balanced_Accuracy"], 4))
    print("Macro Precision/Recall/F1:", round(metrics["Macro_Precision"], 4), round(metrics["Macro_Recall"], 4), round(metrics["Macro_F1"], 4))
    print("ROC AUC:", round(metrics["ROC_AUC"], 4) if metrics["ROC_AUC"] == metrics["ROC_AUC"] else "NaN")
    print("MAE:", round(metrics["MAE"], 4))
    print("Confusion Matrix:")
    print(confusion_matrix_to_pretty_table(metrics["Confusion_Matrix"], labels=class_labels.tolist()))

    return res, metrics


def fit_eval_ordered_logit(
    df: pd.DataFrame,
    y_col: str,
    features: list[str],
    *,
    classes: list[int] | None = None,
    numeric_cols: list[str] | None = None,
    sample_n: int | None = 200_000,
    test_size: float = 0.30,
    random_state: int = 42,
    min_count_rare: int = 3000,
    rare_cols: list[str] | None = None,
    keep_as_is_cols: list[str] | None = None,
    keep_levels: set[str] | None = None,
    method: str = "bfgs",
    maxiter: int = 200,
    title: str = "Ordered Logit",
    link: str = "logit",
) -> tuple[object, dict]:
    
    df_local = df.copy()

    # Coerce specified columns to numeric
    cols_to_numeric = []
    if numeric_cols:
        cols_to_numeric.extend(numeric_cols)
    cols_to_numeric.append(y_col)

    for c in dict.fromkeys(cols_to_numeric):
        if c in df_local.columns:
            df_local[c] = pd.to_numeric(df_local[c], errors="coerce")

    if classes is None:
        classes = [1, 2, 3]

    df_local = df_local[df_local[y_col].isin(classes)].copy()

    # Drop missing rows for modeling columns
    df_local = df_local.dropna(subset=features + [y_col]).copy()

    if df_local.shape[0] == 0:
        # Helpful debug message instead of sklearn's n_samples=0 error
        non_na = df[y_col].notna().sum() if y_col in df.columns else 0
        uniques = df[y_col].astype("string").value_counts().head(10) if y_col in df.columns else None
        raise ValueError(
            f"[fit_eval_ordered_logit] After coercion+filtering, dataset is empty.\n"
            f"- y_col={y_col!r}, classes={classes}\n"
            f"- non-missing y in original df: {non_na}\n"
            f"- top y values (original, as string):\n{uniques}"
        )

    # Group rare categories on selected columns
    if keep_as_is_cols is None:
        keep_as_is_cols = []

    if rare_cols is None:
        numeric_set = set(numeric_cols or [])
        rare_cols = [c for c in features if c not in numeric_set]

    rare_cols = [c for c in rare_cols if c not in set(keep_as_is_cols)]
    for c in rare_cols:
        if c in df_local.columns:
            df_local = group_rare_levels(
                df_local, c, min_count=min_count_rare, other_label="RARE", keep_levels=keep_levels
            )

    # Sample for speed
    if sample_n is not None and df_local.shape[0] > sample_n:
        df_local = df_local.sample(sample_n, random_state=random_state)

    # Build design matrix and align y by index (important!)
    X = build_design_matrix_for_ordinal(df_local, features, drop_first=True)
    y = df_local.loc[X.index, y_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    res, metrics = fit_and_evaluate_ordinal(
        y_train=y_train,
        X_train=X_train,
        y_test=y_test,
        X_test=X_test,
        link=link,
        method=method,
        maxiter=maxiter,
        title=title,
    )

    return res, metrics


__all__ = [
    "group_rare_levels",
    "build_design_matrix",
    "build_design_matrix_for_ordinal",
    "make_or_table",
    "make_or_table_ordinal",
    "make_or_table_ordered",
    "evaluate_binary_classifier",
    "tune_threshold_for_balanced_accuracy",
    "comparison_table",
    "model_comparison_pretty",
    "fit_and_evaluate_ordinal",
    "fit_eval_ordered_logit",
]
