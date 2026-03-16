from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from compare_models import OUTPUT_DIR as MODEL_OUTPUT_DIR
from compare_models import build_model_comparison
from monte_carlo_simulation import OUTPUT_DIR as MONTE_CARLO_OUTPUT_DIR
from monte_carlo_simulation import run_monte_carlo_simulation
from simulation_analysis import OUTPUT_DIR as SIM_OUTPUT_DIR
from simulation_analysis import generate_simulation_outputs

FIGURE_DIR = Path("outputs/figures")

VARIABLE_LABELS = {
    "C_WTHR": "Weather",
    "C_RSUR": "Road surface",
    "C_RALN": "Road alignment",
    "C_TRAF": "Traffic control",
    "C_MNTH": "Month",
    "C_WDAY": "Weekday",
    "C_HOUR": "Hour",
    "C_VEHS": "Vehicles involved",
    "P_SAFE": "Safety device",
    "P_USER": "Road user type",
    "P_SEX": "Sex",
    "V_TYPE": "Vehicle type",
    "VEHICLE_AGE": "Vehicle age",
    "P_AGE": "Person age",
}

VALUE_LABELS = {
    "C_WTHR": {
        "1": "Clear and sunny",
        "2": "Cloudy",
        "3": "Raining",
        "4": "Snowing",
        "5": "Freezing rain / sleet / hail",
        "6": "Visibility limitation",
        "7": "Strong wind",
        "Q": "Other weather condition",
        "OTHER": "Other weather condition",
        "U": "Unknown weather condition",
        "X": "No weather data",
    },
    "C_RSUR": {
        "1": "Dry, normal",
        "2": "Wet",
        "3": "Snow",
        "4": "Slush / wet snow",
        "5": "Icy",
        "6": "Sand / gravel / dirt",
        "7": "Muddy",
        "8": "Oil",
        "9": "Flooded",
        "Q": "Other road surface",
        "OTHER": "Other road surface",
        "U": "Unknown road surface",
        "X": "No road-surface data",
    },
    "C_RALN": {
        "1": "Straight and level",
        "2": "Straight with gradient",
        "3": "Curved and level",
        "4": "Curved with gradient",
        "5": "Top of hill or gradient",
        "6": "Bottom of hill or gradient",
        "Q": "Other road alignment",
        "OTHER": "Other road alignment",
        "U": "Unknown road alignment",
        "X": "No road-alignment data",
    },
    "C_TRAF": {
        "1": "Traffic signals fully operational",
        "2": "Traffic signals in flashing mode",
        "3": "Stop sign",
        "4": "Yield sign",
        "5": "Warning sign",
        "6": "Pedestrian crosswalk",
        "7": "Police officer",
        "8": "School guard / flagman",
        "9": "School crossing",
        "10": "Reduced speed zone",
        "11": "No passing zone sign",
        "12": "Markings on the road",
        "13": "School bus stopped with signal lights flashing",
        "14": "School bus stopped with signal lights not flashing",
        "15": "Railway crossing with signals / gates",
        "16": "Railway crossing with signs only",
        "17": "Control device not specified",
        "18": "No control present",
        "QQ": "Other traffic control",
        "OTHER": "Other traffic control",
        "UU": "Unknown traffic control",
        "XX": "No traffic-control data",
    },
    "P_SAFE": {
        "1": "No safety device used",
        "2": "Safety device used",
        "9": "Helmet worn",
        "10": "Reflective clothing worn",
        "11": "Helmet and reflective clothing used",
        "12": "Other safety device",
        "13": "No safety device equipped",
        "NN": "Not applicable",
        "QQ": "Other safety-device code",
        "OTHER": "Other safety-device code",
        "UU": "Unknown safety-device code",
        "XX": "No safety-device data",
    },
    "P_USER": {
        "1": "Motor vehicle driver",
        "2": "Motor vehicle passenger",
        "3": "Pedestrian",
        "4": "Bicyclist",
        "5": "Motorcyclist",
        "U": "Unknown / not stated road user",
        "OTHER": "Other road user type",
    },
    "P_SEX": {
        "F": "Female",
        "M": "Male",
        "N": "Not applicable",
        "U": "Unknown",
        "X": "No sex data",
        "OTHER": "Other / unknown sex",
    },
    "V_TYPE": {
        "1": "Light-duty vehicle",
        "5": "Panel / cargo van",
        "6": "Other truck / van",
        "7": "Unit truck > 4,536 kg",
        "8": "Road tractor",
        "9": "School bus",
        "10": "Smaller school bus",
        "11": "Urban / intercity bus",
        "14": "Motorcycle / moped",
        "16": "Off-road vehicle",
        "17": "Bicycle",
        "18": "Motorhome",
        "19": "Farm equipment",
        "20": "Construction equipment",
        "21": "Fire engine",
        "22": "Snowmobile",
        "23": "Streetcar",
        "NN": "Not applicable vehicle type",
        "QQ": "Other vehicle type",
        "OTHER": "Other vehicle type",
        "UU": "Unknown vehicle type",
        "XX": "No vehicle-type data",
    },
}

TERM_LABEL_OVERRIDES = {
    "const": "Intercept",
    "P_SEX_M": "Male (vs female)",
}

SEX_LABELS = {
    "F": "Female",
    "M": "Male",
}

RISK_BAND_ORDER = [
    "Below 0.25",
    "0.25 to <0.50",
    "0.50 to <0.75",
    "0.75 and above",
]

SCENARIO_VALUE_LABELS = {
    ("C_TRAF", "1"): "Traffic signals fully operational",
    ("C_TRAF", "18"): "Traffic signal removed",
    ("C_WTHR", "2"): "Overcast / cloudy",
    ("C_WTHR", "6"): "Visibility reduced (fog / smoke / dust / drifting snow)",
    ("C_RSUR", "6"): "Sand / gravel / dirt road surface",
    ("C_RALN", "3"): "Curved and level road",
    ("P_SAFE", "9"): "Helmet worn",
    ("P_USER", "5"): "Motorcyclist",
}


def ensure_figure_dir() -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR


def load_or_generate_model_comparison() -> pd.DataFrame:
    path = MODEL_OUTPUT_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return build_model_comparison(save_output=True)


def load_or_generate_simulation_outputs() -> dict[str, pd.DataFrame]:
    required = {
        "baseline_profile": SIM_OUTPUT_DIR / "baseline_profile.csv",
        "category_switch": SIM_OUTPUT_DIR / "category_switch_simulation.csv",
        "sex_sensitivity": SIM_OUTPUT_DIR / "sex_sensitivity.csv",
        "vehicle_age_sensitivity": SIM_OUTPUT_DIR / "vehicle_age_sensitivity.csv",
        "person_age_sensitivity": SIM_OUTPUT_DIR / "person_age_sensitivity.csv",
        "risk_profiles": SIM_OUTPUT_DIR / "risk_profiles.csv",
        "combined_or_table": SIM_OUTPUT_DIR / "combined_binary_odds_ratios.csv",
    }
    if all(path.exists() for path in required.values()):
        return {key: pd.read_csv(path) for key, path in required.items()}

    generated = generate_simulation_outputs(save_outputs=True)
    return {
        "baseline_profile": generated["baseline_profile"],
        "category_switch": generated["category_switch"],
        "sex_sensitivity": generated["sex_sensitivity"],
        "vehicle_age_sensitivity": generated["vehicle_age_sensitivity"],
        "person_age_sensitivity": generated["person_age_sensitivity"],
        "risk_profiles": generated["risk_profiles"],
        "combined_or_table": generated["combined_or_table"],
    }


def load_or_generate_monte_carlo() -> dict[str, pd.DataFrame]:
    draws_path = MONTE_CARLO_OUTPUT_DIR / "monte_carlo_draws.csv"
    summary_path = MONTE_CARLO_OUTPUT_DIR / "monte_carlo_summary.csv"
    risk_bands_path = MONTE_CARLO_OUTPUT_DIR / "monte_carlo_risk_bands.csv"
    if draws_path.exists() and summary_path.exists() and risk_bands_path.exists():
        return {
            "monte_carlo_draws": pd.read_csv(draws_path),
            "monte_carlo_summary": pd.read_csv(summary_path),
            "risk_bands": pd.read_csv(risk_bands_path),
        }

    generated = run_monte_carlo_simulation(save_outputs=True)
    return {
        "monte_carlo_draws": generated["monte_carlo_draws"],
        "monte_carlo_summary": generated["monte_carlo_summary"],
        "risk_bands": generated["risk_bands"],
    }


def prettify_term_label(term: str, *, width: int = 28) -> str:
    if term in TERM_LABEL_OVERRIDES:
        return fill(TERM_LABEL_OVERRIDES[term], width=width)

    for prefix in sorted(VARIABLE_LABELS, key=len, reverse=True):
        marker = f"{prefix}_"
        if term.startswith(marker):
            code = term[len(marker):]
            mapped_label = lookup_value_label(prefix, code)
            if mapped_label:
                return fill(mapped_label, width=width)
            return fill(f"{VARIABLE_LABELS[prefix]} ({code})", width=width)

    return fill(term.replace("_", " "), width=width)


def prettify_axis_label(raw_label: str) -> str:
    return VARIABLE_LABELS.get(raw_label, raw_label.replace("_", " "))


def normalize_code(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def lookup_value_label(variable: str, value: object) -> str | None:
    label_map = VALUE_LABELS.get(variable)
    if not label_map or pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    candidates: list[str] = [raw, raw.upper()]
    normalized = normalize_code(raw)
    candidates.extend([normalized, normalized.upper()])
    if normalized.isdigit():
        candidates.append(normalized.zfill(2))

    for candidate in dict.fromkeys(candidates):
        label = label_map.get(candidate)
        if label:
            return label

    return None


def format_profile_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "Missing"

    mapped_label = lookup_value_label(column, value)
    if mapped_label:
        return mapped_label

    if isinstance(value, str):
        return value.strip()

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:.1f}"


def prettify_scenario_label(row: pd.Series, *, width: int = 38) -> str:
    variable = str(row.get("Variable", ""))
    scenario_value = normalize_code(row.get("Scenario_Value", ""))

    if row.get("Scenario") == "Baseline (traffic signal)":
        return fill("Baseline (traffic signals fully operational)", width=width)

    label = SCENARIO_VALUE_LABELS.get((variable, scenario_value))
    if label:
        prefix = VARIABLE_LABELS.get(variable, variable.replace("_", " "))
        return fill(f"{prefix}: {label}", width=width)

    mapped_label = lookup_value_label(variable, scenario_value)
    if mapped_label:
        prefix = VARIABLE_LABELS.get(variable, variable.replace("_", " "))
        return fill(f"{prefix}: {mapped_label}", width=width)

    return fill(str(row.get("Scenario", "")), width=width)


def save_model_metric_chart(
    df_model: pd.DataFrame,
    *,
    metric: str,
    title: str,
    filename: str,
) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = df_model.dropna(subset=[metric]).sort_values(metric, ascending=False).copy()
    sns.barplot(data=plot_df, x=metric, y="Model", hue="Family", dodge=False, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("")
    fig.tight_layout()
    path = ensure_figure_dir() / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_odds_ratio_plots(df_or: pd.DataFrame, top_n: int = 12) -> list[Path]:
    sns.set_theme(style="whitegrid")
    output_paths: list[Path] = []

    risk_df = df_or.nlargest(top_n, "Odds_Ratio").sort_values("Odds_Ratio", ascending=True)
    risk_df["Term_Label"] = risk_df["Term"].map(prettify_term_label)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=risk_df, x="Odds_Ratio", y="Term_Label", color="#c44e52", ax=ax)
    ax.set_title("Top Risk-Increasing Odds Ratios")
    ax.set_xlabel("Odds Ratio")
    ax.set_ylabel("Variable level")
    fig.text(
        0.01,
        0.01,
        "Labels use NCDB descriptions where available; unmapped levels keep their original code in parentheses.",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    risk_path = ensure_figure_dir() / "top_risk_odds_ratios.png"
    fig.savefig(risk_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(risk_path)

    protective_df = df_or.nsmallest(top_n, "Odds_Ratio").sort_values("Odds_Ratio", ascending=False)
    protective_df["Term_Label"] = protective_df["Term"].map(prettify_term_label)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=protective_df, x="Odds_Ratio", y="Term_Label", color="#4c72b0", ax=ax)
    ax.set_title("Top Protective Odds Ratios")
    ax.set_xlabel("Odds Ratio")
    ax.set_ylabel("Variable level")
    fig.text(
        0.01,
        0.01,
        "Labels use NCDB descriptions where available; unmapped levels keep their original code in parentheses.",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    protective_path = ensure_figure_dir() / "top_protective_odds_ratios.png"
    fig.savefig(protective_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(protective_path)

    return output_paths


def save_baseline_profile_plot(df_baseline: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    baseline_row = df_baseline.iloc[0]
    profile_columns = [
        col for col in df_baseline.columns
        if col not in {"Predicted_Probability", "CI_Lower_95", "CI_Upper_95"}
    ]
    profile_table = pd.DataFrame(
        {
            "Feature": [prettify_axis_label(col) for col in profile_columns],
            "Value": [format_profile_value(col, baseline_row[col]) for col in profile_columns],
        }
    )

    fig, (ax_table, ax_prob) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    ax_table.axis("off")
    table = ax_table.table(
        cellText=profile_table.values,
        colLabels=profile_table.columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax_table.set_title("Baseline Profile Inputs", pad=12)

    probability = float(baseline_row["Predicted_Probability"])
    lower = float(baseline_row["CI_Lower_95"])
    upper = float(baseline_row["CI_Upper_95"])
    ax_prob.errorbar(
        x=probability,
        y=0,
        xerr=[[probability - lower], [upper - probability]],
        fmt="o",
        color="#2a7f62",
        ecolor="black",
        capsize=4,
        markersize=8,
    )
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(-0.75, 0.75)
    ax_prob.set_yticks([])
    ax_prob.set_xlabel("Predicted Probability")
    ax_prob.set_title("Baseline Risk Estimate", pad=12)
    ax_prob.text(
        probability,
        0.18,
        f"{probability:.1%}\n95% CI {lower:.1%} to {upper:.1%}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout()
    path = ensure_figure_dir() / "baseline_profile_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_probability_curve(df_curve: pd.DataFrame, title: str, filename: str) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_curve["Value"], df_curve["Predicted_Probability"], color="#2a7f62", linewidth=2)
    if {"CI_Lower_95", "CI_Upper_95"}.issubset(df_curve.columns):
        ax.fill_between(
            df_curve["Value"],
            df_curve["CI_Lower_95"],
            df_curve["CI_Upper_95"],
            color="#2a7f62",
            alpha=0.2,
        )
    ax.set_title(title)
    ax.set_xlabel(prettify_axis_label(str(df_curve["Variable"].iloc[0])))
    ax.set_ylabel("Predicted Probability")
    fig.tight_layout()
    path = ensure_figure_dir() / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_risk_profile_plot(df_profiles: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    plot_df = df_profiles.sort_values("Predicted_Probability", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=plot_df,
        x="Predicted_Probability",
        y="Profile",
        color="#dd8452",
        ax=ax,
    )

    xerr = [
        plot_df["Predicted_Probability"] - plot_df["CI_Lower_95"],
        plot_df["CI_Upper_95"] - plot_df["Predicted_Probability"],
    ]
    ax.errorbar(
        x=plot_df["Predicted_Probability"],
        y=range(len(plot_df)),
        xerr=xerr,
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )

    ax.set_title("Risk Profile Comparison")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("")
    fig.tight_layout()
    path = ensure_figure_dir() / "risk_profile_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_scenario_comparison_plot(df_scenarios: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = df_scenarios.copy()
    plot_df["Scenario_Label"] = plot_df.apply(prettify_scenario_label, axis=1)
    sns.barplot(
        data=plot_df,
        x="Predicted_Probability",
        y="Scenario_Label",
        hue="Scenario_Label",
        palette="crest",
        legend=False,
        ax=ax,
    )

    if {"CI_Lower_95", "CI_Upper_95"}.issubset(plot_df.columns):
        xerr = [
            plot_df["Predicted_Probability"] - plot_df["CI_Lower_95"],
            plot_df["CI_Upper_95"] - plot_df["Predicted_Probability"],
        ]
        ax.errorbar(
            x=plot_df["Predicted_Probability"],
            y=range(len(plot_df)),
            xerr=xerr,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )

    ax.set_title("Scenario Comparison")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("")
    fig.tight_layout()
    path = ensure_figure_dir() / "scenario_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_risk_tornado_chart(df_scenarios: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    plot_df = df_scenarios.copy()
    plot_df = plot_df[plot_df["Scenario"] != "Baseline (traffic signal)"].copy()
    plot_df["Scenario_Label"] = plot_df.apply(prettify_scenario_label, axis=1)
    plot_df = plot_df.sort_values("Absolute_Change", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4c72b0" if value < 0 else "#c44e52" for value in plot_df["Absolute_Change"]]
    ax.barh(plot_df["Scenario_Label"], plot_df["Absolute_Change"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Risk Tornado Chart")
    ax.set_xlabel("Absolute Change in Predicted Probability vs Baseline")
    ax.set_ylabel("")
    fig.tight_layout()
    path = ensure_figure_dir() / "risk_tornado_chart.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_sex_sensitivity_plot(df_sex: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    plot_df = df_sex.copy()
    plot_df["Sex_Label"] = plot_df["P_SEX"].map(lambda value: SEX_LABELS.get(str(value), str(value)))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=plot_df,
        x="Sex_Label",
        y="Predicted_Probability",
        hue="Sex_Label",
        palette="mako",
        legend=False,
        ax=ax,
    )

    xerr = [
        plot_df["Predicted_Probability"] - plot_df["CI_Lower_95"],
        plot_df["CI_Upper_95"] - plot_df["Predicted_Probability"],
    ]
    ax.errorbar(
        x=range(len(plot_df)),
        y=plot_df["Predicted_Probability"],
        yerr=xerr,
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )

    ax.set_title("Sex Sensitivity Comparison")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Predicted Probability")
    fig.tight_layout()
    path = ensure_figure_dir() / "sex_sensitivity_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_monte_carlo_risk_bands_plot(df_risk_bands: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    plot_df = df_risk_bands.copy()
    plot_df["Risk_Band"] = pd.Categorical(plot_df["Risk_Band"], categories=RISK_BAND_ORDER, ordered=True)
    plot_df = plot_df.sort_values("Risk_Band")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Risk_Band", y="Share", color="#4c72b0", ax=ax)
    ax.set_title("Monte Carlo Risk Bands")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Draws")
    ax.set_ylim(0, max(0.1, float(plot_df["Share"].max()) * 1.15))
    ax.tick_params(axis="x", rotation=12)

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            idx,
            float(row.Share) + 0.01,
            f"{row.Share:.1%}\n(n={int(row.Count)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    path = ensure_figure_dir() / "monte_carlo_risk_bands.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_monte_carlo_distribution_plot(draws: pd.DataFrame, summary: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(draws["Predicted_Probability"], bins=40, kde=True, color="#8172b3", ax=ax)
    summary_row = summary.iloc[0]
    ax.axvline(
        float(summary_row["Baseline_Probability"]),
        color="#c44e52",
        linestyle="--",
        linewidth=2,
        label=f"Baseline = {float(summary_row['Baseline_Probability']):.3f}",
    )
    ax.axvline(
        float(summary_row["mean_probability"]),
        color="#2a7f62",
        linestyle="-.",
        linewidth=2,
        label=f"Mean = {float(summary_row['mean_probability']):.3f}",
    )
    ax.axvline(
        0.75,
        color="#222222",
        linestyle=":",
        linewidth=2,
        label="High-risk cutoff = 0.75",
    )
    ax.set_title("Monte Carlo Predicted Risk Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = ensure_figure_dir() / "monte_carlo_risk_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    model_df = load_or_generate_model_comparison()
    simulation_outputs = load_or_generate_simulation_outputs()
    monte_carlo_outputs = load_or_generate_monte_carlo()

    save_model_metric_chart(
        model_df,
        metric="ROC_AUC",
        title="Model Comparison by ROC AUC",
        filename="model_comparison_roc_auc.png",
    )
    save_model_metric_chart(
        model_df,
        metric="Balanced_Accuracy",
        title="Model Comparison by Balanced Accuracy",
        filename="model_comparison_balanced_accuracy.png",
    )
    save_baseline_profile_plot(simulation_outputs["baseline_profile"])
    save_odds_ratio_plots(simulation_outputs["combined_or_table"])
    save_scenario_comparison_plot(simulation_outputs["category_switch"])
    save_risk_tornado_chart(simulation_outputs["category_switch"])
    save_risk_profile_plot(simulation_outputs["risk_profiles"])
    save_probability_curve(
        simulation_outputs["vehicle_age_sensitivity"],
        title="Vehicle Age vs Predicted Injury Probability",
        filename="vehicle_age_probability_curve.png",
    )
    save_probability_curve(
        simulation_outputs["person_age_sensitivity"],
        title="Person Age vs Predicted Injury Probability",
        filename="person_age_probability_curve.png",
    )
    save_monte_carlo_distribution_plot(
        monte_carlo_outputs["monte_carlo_draws"],
        monte_carlo_outputs["monte_carlo_summary"],
    )
    save_monte_carlo_risk_bands_plot(monte_carlo_outputs["risk_bands"])
    save_sex_sensitivity_plot(simulation_outputs["sex_sensitivity"])

    print(f"\nSaved figure outputs to: {ensure_figure_dir().resolve()}")


if __name__ == "__main__":
    main()
