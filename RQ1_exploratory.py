# rq1_exploratory_collision.py
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from Load_prepare import df, RANDOM_STATE, N_SAMPLE_RQ1, MIN_COUNT_RARE
from utils import group_rare_levels, build_design_matrix, evaluate_binary_classifier, tune_threshold_for_balanced_accuracy

def run_rq1_exploratory(alpha=0.05):
    print("RQ1: collision-level EXPLORATORY model")

    df_rq1 = df[df["C_SEV"].isin([1, 2])].copy()
    df_rq1["Y_FATAL_COLLISION"] = (df_rq1["C_SEV"] == 1).astype(int)

    print("\nCollision severity distribution (RQ1):")
    print(df_rq1["Y_FATAL_COLLISION"].value_counts())
    print(df_rq1["Y_FATAL_COLLISION"].value_counts(normalize=True))

    rq1_features = ["C_WTHR","C_RSUR","C_RALN","C_TRAF","C_MNTH","C_WDAY","C_HOUR","C_VEHS"]
    df_rq1 = df_rq1.dropna(subset=rq1_features + ["Y_FATAL_COLLISION"]).copy()

    for c in ["C_WTHR", "C_RSUR", "C_RALN", "C_TRAF"]:
        df_rq1 = group_rare_levels(df_rq1, c, min_count=MIN_COUNT_RARE, other_label="RARE")

    # sample for speed (your setting)
    df_rq1 = df_rq1.sample(min(N_SAMPLE_RQ1, len(df_rq1)), random_state=RANDOM_STATE).copy()

    y = df_rq1["Y_FATAL_COLLISION"].astype(int)
    X = build_design_matrix(df_rq1, rq1_features, drop_first=True, add_constant=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    print("\nRQ1 shapes:", X_train.shape, X_test.shape)

    # Regularized Logit
    model = sm.Logit(y_train, X_train)
    res = model.fit_regularized(method="l1", alpha=0.1, maxiter=1000)

    nonzero = int((res.params != 0).sum())
    print("\nRQ1 Regularized model fitted successfully.")
    print("Non-zero coefficients:", nonzero, "out of", len(res.params))

    p_test = res.predict(X_test)

    # Default threshold
    r_default = evaluate_binary_classifier(y_test, p_test, threshold=0.50, title="RQ1 Exploratory (0.50)")

    # Tune threshold
    best_t, best_bacc = tune_threshold_for_balanced_accuracy(
        y_test, p_test, thresholds=np.linspace(0.01, 0.50, 50)
    )
    print(f"\nBest threshold for balanced accuracy: {best_t:.3f} (BAcc={best_bacc:.4f})")
    r_tuned = evaluate_binary_classifier(y_test, p_test, threshold=best_t, title="RQ1 Exploratory (tuned)")

    # Top coefficients
    coef = res.params.copy()
    coef = coef[coef != 0].sort_values(key=np.abs, ascending=False)
    print("\nTop non-zero coefficients (by absolute value):")
    print(coef.head(20))

    return res, r_default, r_tuned

if __name__ == "__main__":
    run_rq1_exploratory()
