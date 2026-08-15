import math

import numpy as np
from numpy.typing import NDArray

from algebrandom.core.primitive import RandomVariable


class StudentT(RandomVariable):

    df: float
    mu: float
    sigma: float

    def __init__(self, df: float, mu: float = 0.0, sigma: float = 1.0, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if df <= 0:
            raise ValueError("df must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.df = df
        self.mu = mu
        self.sigma = sigma
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return self.mu + self.sigma * rng.standard_t(self.df, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        dist = stats.t(df=self.df, loc=self.mu, scale=self.sigma)
        return dist.cdf(y) - dist.cdf(x)

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.t(df=self.df, loc=self.mu, scale=self.sigma).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.t(df=self.df, loc=self.mu, scale=self.sigma).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        k = int(k)
        if self.df <= k:
            raise ValueError("Student-t moment undefined for df <= k")
        from scipy import stats
        return float(stats.t(df=self.df, loc=self.mu, scale=self.sigma).moment(k))

    @property
    def name(self) -> str:
        return "StudentT" + "[" + super().name + "]" + "(" + str(self.df) + "," + str(self.mu) + "," + str(self.sigma) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.identities import StudentTIdentityPattern
        from algebrandom.patterns.studentt.affine import StudentTScalingPattern, StudentTShiftPattern, StudentTSquarePattern

        return [
            StudentTIdentityPattern(),
            StudentTSquarePattern(),
            StudentTScalingPattern(),
            StudentTShiftPattern(),
        ]



def specialize_studentt(df: float, mu: float, sigma: float, name: str = "", expression=None) -> RandomVariable:
    from algebrandom.instances.cauchy import Cauchy

    if math.isclose(df, 1.0):
        return Cauchy(mu, sigma, name=name, expression=expression)
    return StudentT(df, mu, sigma, name=name, expression=expression)
