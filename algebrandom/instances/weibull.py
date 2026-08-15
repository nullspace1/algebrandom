import math

import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class Weibull(RandomVariable):

    shape: float
    scale: float

    def __init__(self, shape: float, scale: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if shape <= 0:
            raise ValueError("shape must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.shape = shape
        self.scale = scale
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return self.scale * rng.weibull(self.shape, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.weibull_min(c=self.shape, scale=self.scale)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.weibull_min(c=self.shape, scale=self.scale).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.weibull_min(c=self.shape, scale=self.scale).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        from scipy import special
        k = int(k)
        return float(self.scale**k * special.gamma(1 + k / self.shape))

    @property
    def name(self) -> str:
        return "Weibull" + "[" + super().name + "]" + "(" + str(self.shape) + "," + str(self.scale) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.identities import WeibullIdentityPattern
        from algebrandom.patterns.weibull.transforms import WeibullPowerPattern, WeibullScalingPattern

        return [
            WeibullIdentityPattern(),
            WeibullPowerPattern(),
            WeibullScalingPattern(),
        ]



def specialize_weibull(shape: float, scale: float, name: str = "", expression=None) -> RandomVariable:
    from algebrandom.instances.exponential import Exponential

    if math.isclose(shape, 1.0):
        return Exponential(1 / scale, name=name, expression=expression)
    return Weibull(shape, scale, name=name, expression=expression)
