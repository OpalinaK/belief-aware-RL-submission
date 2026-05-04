import json

import numpy as np

try:
    from pomegranate import GeneralMixtureModel  # legacy API
except Exception:  # pragma: no cover - runtime fallback for newer pomegranate
    GeneralMixtureModel = None


order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [2.9, 1.2],
            "frozen": False,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [100.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [200.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [300.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [400.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [500.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [600.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [700.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [800.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [900.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1000.0, 0.15],
            "frozen": True,
        },
    ],
    "weights": [
        0.2,
        0.7,
        0.06,
        0.004,
        0.0329,
        0.001,
        0.0006,
        0.0004,
        0.0005,
        0.0003,
        0.0003,
    ],
}


class OrderSizeModel:
    def __init__(self) -> None:
        self.model = (
            GeneralMixtureModel.from_json(json.dumps(order_size))
            if GeneralMixtureModel is not None
            else None
        )

    def sample(self, random_state: np.random.RandomState) -> float:
        if self.model is not None:
            return round(self.model.sample(random_state=random_state))

        # Fallback sampler when pomegranate's legacy GeneralMixtureModel is absent.
        weights = np.array(order_size["weights"], dtype=float)
        weights /= weights.sum()
        idx = random_state.choice(len(weights), p=weights)
        dist = order_size["distributions"][idx]
        mu, sigma = dist["parameters"]
        if dist["name"] == "LogNormalDistribution":
            return round(float(random_state.lognormal(mean=mu, sigma=sigma)))
        return round(float(random_state.normal(loc=mu, scale=sigma)))
