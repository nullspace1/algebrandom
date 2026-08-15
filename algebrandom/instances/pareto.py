import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class Pareto(RandomVariable):

    xmin: float
    shape: float

    def __init__(self, xmin: float, shape: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if xmin <= 0:
            raise ValueError("xmin must be positive")
        if shape <= 0:
            raise ValueError("shape must be positive")
        self.xmin = xmin
        self.shape = shape
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return self.xmin * (1 + rng.pareto(self.shape, count))

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.pareto(b=self.shape, scale=self.xmin)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.pareto(b=self.shape, scale=self.xmin).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.pareto(b=self.shape, scale=self.xmin).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        k = int(k)
        if self.shape <= k:
            raise ValueError("Pareto moment undefined for shape <= k")
        return self.shape * self.xmin**k / (self.shape - k)

    @property
    def name(self) -> str:
        return "Pareto" + "[" + super().name + "]" + "(" + str(self.xmin) + "," + str(self.shape) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.pareto.transforms import ParetoLogPattern, ParetoPowerPattern, ParetoScalingPattern

        return [
            ParetoLogPattern(),
            ParetoPowerPattern(),
            ParetoScalingPattern(),
        ]

