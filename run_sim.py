from town_model import TownModel


def run_sim():
    # Create the model
    model = TownModel(N=200, policy_mode="targeted", seed=42)

    # Simulate X steps (e.g., 60 days * 24 hours)
    steps = 24 * 60
    for _ in range(steps):
        model.step()

    # Collect results
    results = model.datacollector.get_model_vars_dataframe()
    print(results.head())
    print(results.tail())

    return results


def plot_results(results):
    import matplotlib.pyplot as plt

    hours = results.index

    fig, ax1 = plt.subplots()
    color_inf = "tab:red"
    color_stress = "tab:blue"

    ax1.set_xlabel("Hour of simulation")
    ax1.set_ylabel("Infected agents", color=color_inf)
    ax1.plot(hours, results["Infected"], color=color_inf, label="Infected")
    ax1.tick_params(axis="y", labelcolor=color_inf)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg stress (0-1)", color=color_stress)
    ax2.plot(hours, results["AvgStress"], color=color_stress, label="Avg Stress")
    ax2.tick_params(axis="y", labelcolor=color_stress)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Pandemic trajectory vs. mental stress")
    fig.tight_layout()
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output = "simulation_plot.png"
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    results = run_sim()
    plot_results(results)
