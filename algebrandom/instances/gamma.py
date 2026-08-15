import math

import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class Gamma(RandomVariable):

    shape: float
    rate: float

    def __init__(self, shape: float, rate: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if shape <= 0:
            raise ValueError("shape must be positive")
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.shape = shape
        self.rate = rate
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return rng.gamma(self.shape, 1 / self.rate, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.gamma(a=self.shape, scale=1 / self.rate)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.gamma(a=self.shape, scale=1 / self.rate).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.gamma(a=self.shape, scale=1 / self.rate).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        k = int(k)
        result = 1.0
        for i in range(k):
            result *= (self.shape + i) / self.rate
        return result

    @property
    def name(self) -> str:
        return "Gamma" + "[" + super().name + "]" + "(" + str(self.shape) + "," + str(self.rate) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.f.ratio import FRatioPattern
        from algebrandom.patterns.gamma.addition import GammaAdditionPattern
        from algebrandom.patterns.gamma.ratio import GammaRatioPattern
        from algebrandom.patterns.gamma.scaling import GammaScalingPattern
        from algebrandom.patterns.identities import GammaIdentityPattern
        from algebrandom.patterns.studentt.ratio import StudentTRatioPattern

        return [
            GammaIdentityPattern(),
            StudentTRatioPattern(),
            FRatioPattern(),
            GammaRatioPattern(),
            GammaScalingPattern(),
            GammaAdditionPattern(),
        ]



def specialize_gamma(shape: float, rate: float, name: str = "", expression=None) -> RandomVariable:
    from algebrandom.instances.chisquare import ChiSquare
    from algebrandom.instances.exponential import Exponential

    if math.isclose(shape, 1.0):
        return Exponential(rate, name=name, expression=expression)
    df = shape * 2
    if math.isclose(rate, 0.5) and math.isclose(df, round(df)) and round(df) >= 1:
        return ChiSquare(int(round(df)), name=name, expression=expression)
    return Gamma(shape, rate, name=name, expression=expression)
