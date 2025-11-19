"""
Mesa web visualization server for the TownModel.
Run with `python viz.py` (after installing mesa[visualization] dependencies).
"""

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, NetworkModule
try:
    from mesa.visualization.UserParam import UserSettableParameter as _UserSettableParameter
    _legacy_userparam = None
except ImportError:
    _UserSettableParameter = None
    from mesa.visualization import UserParam as _legacy_userparam

from town_model import TownModel


# --- Network portrayal ----------------------------------------------------- #
def network_portrayal(G):
    """
    Render a simple contact network each step where node size reflects stress
    and node color reflects infection state.
    """
    portrayal = {"nodes": [], "edges": []}

    if G is None:
        return portrayal

    for node_id, agent in G.nodes.data("agent"):
        if agent is None:
            continue

        color = "green"
        if agent.health_state == "E":
            color = "orange"
        elif agent.health_state == "I":
            color = "red"
        elif agent.health_state == "R":
            color = "gray"

        portrayal["nodes"].append(
            {
                "id": node_id,
                "size": 3 + 6 * agent.stress,
                "color": color,
                "tooltip": f"{agent.role.title()} stress={agent.stress:.2f}",
            }
        )

    for source, target in G.edges():
        portrayal["edges"].append({"source": source, "target": target})

    return portrayal


# --- Charts ---------------------------------------------------------------- #
infection_chart = ChartModule(
    [
        {"Label": "Infected", "Color": "red"},
    ],
    data_collector_name="datacollector",
)

stress_chart = ChartModule(
    [
        {"Label": "AvgStress", "Color": "blue"},
        {"Label": "HighStressFrac", "Color": "green"},
    ],
    data_collector_name="datacollector",
)


# --- User param compatibility ---------------------------------------------- #


def _make_user_param(param_type, name, *args, **kwargs):
    if _UserSettableParameter is not None:
        return _UserSettableParameter(param_type, name, *args, **kwargs)

    if _legacy_userparam is None:
        raise ImportError("Mesa UserSettableParameter not available")

    if param_type == "slider":
        if len(args) < 4:
            raise ValueError("Slider requires value, min, max, step")
        value, min_value, max_value, step = args[:4]
        return _legacy_userparam.Slider(
            name=name,
            value=value,
            min_value=min_value,
            max_value=max_value,
            step=step,
            description=kwargs.get("description"),
        )
    if param_type == "choice":
        value = kwargs.get("value") if "value" in kwargs else args[0]
        choices = kwargs.get("choices")
        return _legacy_userparam.Choice(
            name=name,
            value=value,
            choices=choices,
            description=kwargs.get("description"),
        )
    if param_type == "checkbox":
        value = args[0] if args else kwargs.get("value")
        return _legacy_userparam.Checkbox(
            name=name,
            value=value,
            description=kwargs.get("description"),
        )

    raise ValueError(f"Unsupported user parameter type: {param_type}")


def slider_param(name, value, min_value, max_value, step):
    return _make_user_param("slider", name, value, min_value, max_value, step)


def choice_param(name, value, choices):
    return _make_user_param("choice", name, value=value, choices=choices)


def checkbox_param(name, value):
    return _make_user_param("checkbox", name, value)


# --- Parameters ------------------------------------------------------------ #
model_params = {
    "N": slider_param("Population Size", 200, 50, 300, 10),
    "policy_mode": choice_param("Policy Mode", "none", ["none", "targeted", "full"]),
    "beta": slider_param("Infection Rate β", 0.05, 0.01, 0.1, 0.01),
    "stress_decay": slider_param("Stress Decay Rate", 0.001, 0.0005, 0.01, 0.0005),
    "incubation_days": slider_param("Incubation (days)", 3, 1, 5, 1),
    "infectious_days": slider_param("Infectious (days)", 7, 3, 14, 1),
    "enable_owner_econ_stress": checkbox_param("Owners feel economic stress?", True),
    "enable_social_influence": checkbox_param("Enable social influence stress?", True),
}


network = NetworkModule(network_portrayal, 500, 500)


server = ModularServer(
    TownModel,
    [network, infection_chart, stress_chart],
    "Pandemic Social Simulation",
    model_params,
)
server.port = 8521


if __name__ == "__main__":
    server.launch()
