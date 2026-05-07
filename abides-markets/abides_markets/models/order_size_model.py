import numpy as np

# Mixture weights and component parameters matching the original pomegranate model.
# Component 0: LogNormal(mu=2.9, sigma=1.2)
# Components 1-10: Normal(mean, std=0.15) at 100, 200, ..., 1000
_WEIGHTS = np.array([0.2, 0.7, 0.06, 0.004, 0.0329, 0.001, 0.0006, 0.0004, 0.0005, 0.0003, 0.0003])
_WEIGHTS /= _WEIGHTS.sum()  # normalise to sum exactly to 1


class OrderSizeModel:
    def sample(self, random_state: np.random.RandomState) -> float:
        component = random_state.choice(len(_WEIGHTS), p=_WEIGHTS)
        if component == 0:
            return round(random_state.lognormal(mean=2.9, sigma=1.2))
        mean = 100.0 * component
        return round(random_state.normal(loc=mean, scale=0.15))
