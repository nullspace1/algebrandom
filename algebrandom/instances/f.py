import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class F(RandomVariable):

    dfn: float
    dfd: float

    def __init__(self, dfn: float, dfd: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if dfn <= 0 or dfd <= 0:
            raise ValueError("degrees of freedom must be positive")
        self.dfn = dfn
        self.dfd = dfd
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return rng.f(self.dfn, self.dfd, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.f(dfn=self.dfn, dfd=self.dfd)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.f(dfn=self.dfn, dfd=self.dfd).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.f(dfn=self.dfn, dfd=self.dfd).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        k = int(k)
        if self.dfd <= 2 * k:
            raise ValueError("F moment undefined for dfd <= 2k")
        from scipy import stats
        return float(stats.f(dfn=self.dfn, dfd=self.dfd).moment(k))

    @property
    def name(self) -> str:
        return "F" + "[" + super().name + "]" + "(" + str(self.dfn) + "," + str(self.dfd) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.f.ratio import FRatioPattern
        from algebrandom.patterns.f.transforms import FReciprocalPattern, FToBetaPattern

        return [
            FRatioPattern(),
            FReciprocalPattern(),
            FToBetaPattern(),
        ]

