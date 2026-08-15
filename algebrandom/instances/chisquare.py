import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class ChiSquare(RandomVariable):

    df: float

    def __init__(self, df: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if df <= 0:
            raise ValueError("df must be positive")
        self.df = df
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return rng.chisquare(self.df, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.chi2(df=self.df)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.chi2(df=self.df).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.chi2(df=self.df).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        k = int(k)
        result = 1.0
        for i in range(k):
            result *= (self.df + 2 * i)
        return result

    @property
    def name(self) -> str:
        return "ChiSquare" + "[" + super().name + "]" + "(" + str(self.df) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.chisquare.addition import ChiSquareAdditionPattern
        from algebrandom.patterns.chisquare.scaling import ChiSquareScalingPattern
        from algebrandom.patterns.f.ratio import FRatioPattern
        from algebrandom.patterns.gamma.ratio import GammaRatioPattern
        from algebrandom.patterns.studentt.ratio import StudentTRatioPattern

        return [
            StudentTRatioPattern(),
            FRatioPattern(),
            GammaRatioPattern(),
            ChiSquareScalingPattern(),
            ChiSquareAdditionPattern(),
        ]

