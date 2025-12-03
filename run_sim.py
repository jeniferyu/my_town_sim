from pathlib import Path
from typing import Tuple

import pandas as pd

from town_model import TownModel


SIM_HOURS = 24 * 60
POLICIES = ["none", "targeted", "full"]
PLOTS_DIR = Path("plots")


def run_policy(policy_mode: str, seed: int = 42) -> pd.DataFrame:
    """Run a single policy scenario and return the resulting dataframe."""

    model = TownModel(N=200, policy_mode=policy_mode, seed=seed)
    for _ in range(SIM_HOURS):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    df = df.reset_index().rename(columns={"index": "hour"})
    df["policy"] = policy_mode
    return df


def summarize_policy(df: pd.DataFrame) -> dict:
    """Compute key summary metrics that help compare policy runs."""

    baseline_stress = df.loc[df["hour"] == 0, "AvgStress"].iloc[0]
    df_sorted = df.sort_values("hour")
    peak_hour_idx = df_sorted["Infected"].idxmax()
    time_to_peak = int(df_sorted.loc[peak_hour_idx, "hour"])

    recovery_hours = None
    tolerance = 0.01
    post_peak = df_sorted[df_sorted["hour"] > time_to_peak]
    recovered = post_peak.loc[post_peak["AvgStress"] <= baseline_stress + tolerance]
    if not recovered.empty:
        recovery_hours = int(recovered.iloc[0]["hour"])

    metrics = {
        "policy": df_sorted["policy"].iloc[0],
        "peak_infected": int(df_sorted["Infected"].max()),
        "time_to_peak_infected_hours": time_to_peak,
        "peak_stress": float(df_sorted["AvgStress"].max()),
        "stress_area_under_curve": float(df_sorted["AvgStress"].sum()),
        "max_owner_stress": float(df_sorted["AvgStress_Owner"].max()),
        "time_until_stress_near_baseline_hours": recovery_hours,
        "max_noncompliant_frac": float(df_sorted["NonCompliantFrac"].max()),
    }
    return metrics


def run_all_policies() -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    summaries = []
    for policy in POLICIES:
        df = run_policy(policy)
        print(f"Policy '{policy}' finished: head/tail preview")
        print(df.head())
        print(df.tail())
        results.append(df)
        summaries.append(summarize_policy(df))

    combined = pd.concat(results, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    return combined, summary_df


def save_or_show(fig, filename: str):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    path = PLOTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"Plot saved to {path}")
    backend = plt.get_backend().lower()
    if "agg" in backend:
        return
    plt.show()


def plot_infections(all_results: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for policy, grp in all_results.groupby("policy"):
        ax.plot(grp["hour"], grp["Infected"], label=policy.title())

    ax.set_xlabel("Hour of simulation")
    ax.set_ylabel("Infected agents")
    ax.set_title("Infection trajectories by policy")
    ax.legend()
    save_or_show(fig, "infection_trajectories.png")


def plot_average_stress(all_results: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for policy, grp in all_results.groupby("policy"):
        ax.plot(grp["hour"], grp["AvgStress"], label=policy.title())

    ax.set_xlabel("Hour of simulation")
    ax.set_ylabel("Avg stress (0-1)")
    ax.set_title("Average stress by policy")
    ax.legend()
    save_or_show(fig, "average_stress.png")


def plot_high_stress_fraction(all_results: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for policy, grp in all_results.groupby("policy"):
        ax.plot(grp["hour"], grp["HighStressFrac"], label=policy.title())

    ax.set_xlabel("Hour of simulation")
    ax.set_ylabel("Fraction with stress > 0.8")
    ax.set_ylim(0, 1)
    ax.set_title("High-stress population share")
    ax.legend()
    save_or_show(fig, "high_stress_fraction.png")


def plot_stress_by_role(all_results: pd.DataFrame):
    import matplotlib.pyplot as plt

    role_cols = {
        "Students": "AvgStress_Student",
        "Office": "AvgStress_Office",
        "Service": "AvgStress_Service",
        "Owners": "AvgStress_Owner",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for ax, (role_name, col) in zip(axes, role_cols.items()):
        for policy, grp in all_results.groupby("policy"):
            ax.plot(grp["hour"], grp[col], label=policy.title())
        ax.set_title(role_name)
        ax.set_ylabel("Stress")
        ax.set_ylim(0, 1.05)

    axes[-2].set_xlabel("Hour")
    axes[-1].set_xlabel("Hour")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Stress by role and policy")
    plt.subplots_adjust(top=0.9, bottom=0.2)
    save_or_show(fig, "stress_by_role.png")


def plot_infection_stress_per_policy(all_results: pd.DataFrame):
    """Create one infection-vs-stress plot per policy with twin axes."""

    import matplotlib.pyplot as plt

    for policy, grp in all_results.groupby("policy"):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Hour of simulation")
        ax1.set_ylabel("Infected agents", color="tab:red")
        ax1.plot(grp["hour"], grp["Infected"], color="tab:red", label="Infected")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg stress", color="tab:blue")
        ax2.plot(grp["hour"], grp["AvgStress"], color="tab:blue", label="Avg Stress")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax1.set_title(f"{policy.title()} policy: infection vs stress")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        save_or_show(fig, f"{policy}_infection_vs_stress.png")


def plot_policy_comparison_small_multiples(all_results: pd.DataFrame):
    """Side-by-side twin-axis plots for each policy in one figure."""

    import matplotlib.pyplot as plt

    policies = sorted(all_results["policy"].unique())
    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 4), sharey=False)
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        grp = all_results[all_results["policy"] == policy]
        ax.set_title(policy.title())
        ax.set_xlabel("Hour")
        ax.set_ylabel("Infected", color="tab:red")
        ax.plot(grp["hour"], grp["Infected"], color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax.twinx()
        ax2.set_ylabel("Avg stress", color="tab:blue")
        ax2.plot(grp["hour"], grp["AvgStress"], color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Policy comparison: infection vs stress")
    save_or_show(fig, "policy_comparison_infection_vs_stress.png")


def main():
    all_results, summary_df = run_all_policies()
    print("\nSummary metrics:")
    print(summary_df)

    plot_infections(all_results)
    plot_average_stress(all_results)
    plot_high_stress_fraction(all_results)
    plot_stress_by_role(all_results)
    plot_infection_stress_per_policy(all_results)
    plot_policy_comparison_small_multiples(all_results)

    summary_df.to_csv("policy_metrics.csv", index=False)
    print("Summary metrics saved to policy_metrics.csv")


if __name__ == "__main__":
    main()
