import math

import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class Exponential(RandomVariable):

    rate: float

    def __init__(self, rate: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return rng.exponential(scale=1 / self.rate, size=count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.expon(scale=1 / self.rate)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.expon(scale=1 / self.rate).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.expon(scale=1 / self.rate).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        return math.factorial(int(k)) / self.rate**int(k)

    @property
    def name(self) -> str:
        return "Exponential" + "[" + super().name + "]" + "(" + str(self.rate) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.exponential.addition import ExponentialAdditionPattern
        from algebrandom.patterns.exponential.scaling import ExponentialScalingPattern
        from algebrandom.patterns.gamma.addition import GammaAdditionPattern
        from algebrandom.patterns.gamma.ratio import GammaRatioPattern
        from algebrandom.patterns.pareto.transforms import ExpToParetoPattern
        from algebrandom.patterns.weibull.transforms import ExponentialPowerWeibullPattern

        return [
            ExpToParetoPattern(),
            ExponentialPowerWeibullPattern(),
            GammaRatioPattern(),
            ExponentialScalingPattern(),
            ExponentialAdditionPattern(),
            GammaAdditionPattern(),
        ]

