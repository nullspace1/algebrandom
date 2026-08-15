import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class Cauchy(RandomVariable):

    x0: float
    gamma: float

    def __init__(self, x0: float, gamma: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.x0 = x0
        self.gamma = gamma
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return self.x0 + self.gamma * rng.standard_cauchy(count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.cauchy(loc=self.x0, scale=self.gamma)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.cauchy(loc=self.x0, scale=self.gamma).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.cauchy(loc=self.x0, scale=self.gamma).cdf(x)

    def moment(self, k: int) -> float:
        raise ValueError("Cauchy moments are undefined")

    @property
    def name(self) -> str:
        return "Cauchy" + "[" + super().name + "]" + "(" + str(self.x0) + "," + str(self.gamma) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.cauchy.addition import CauchyAdditionPattern
        from algebrandom.patterns.cauchy.scaling import CauchyScalingPattern
        from algebrandom.patterns.cauchy.shift import CauchyShiftPattern

        return [
            CauchyScalingPattern(),
            CauchyShiftPattern(),
            CauchyAdditionPattern(),
        ]

