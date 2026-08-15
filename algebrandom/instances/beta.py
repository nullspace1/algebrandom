import numpy as np
from numpy.typing import NDArray

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable


class Beta(RandomVariable):

    alpha: float
    beta: float

    def __init__(self, alpha: float, beta: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return rng.beta(self.alpha, self.beta, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.beta(a=self.alpha, b=self.beta)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.beta(a=self.alpha, b=self.beta).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.beta(a=self.alpha, b=self.beta).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        from scipy import special
        k = int(k)
        return float(special.beta(self.alpha + k, self.beta) / special.beta(self.alpha, self.beta))

    @property
    def name(self) -> str:
        return "Beta" + "[" + super().name + "]" + "(" + str(self.alpha) + "," + str(self.beta) + ")"

    @classmethod
    def _build_patterns(cls) -> list["Pattern"]:
        from algebrandom.patterns.beta.complement import BetaComplementPattern

        return [BetaComplementPattern()]



def specialize_beta(alpha: float, beta: float, name: str = "", expression=None) -> RandomVariable:
    import math

    from algebrandom.instances.uniform import Uniform

    if math.isclose(alpha, 1.0) and math.isclose(beta, 1.0):
        return Uniform(0.0, 1.0, name=name, expression=expression)
    return Beta(alpha, beta, name=name, expression=expression)
