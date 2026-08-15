import math
from typing import cast

import numpy as np
from numpy.typing import NDArray
import sympy as sp
from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable




class Normal(RandomVariable):
    mu: float
    sigma: float

    def __init__(self, mu: float, sigma: float, name: str = "", default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        if sigma <= 0:
            raise ValueError("std must be positive")
        self.mu = mu
        self.sigma = sigma
        super().__init__(name, default_sample_count, rng, **kwargs)

    def _sample_no_cache(self, count: int, rng: np.random.Generator , cache: dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return rng.normal(self.mu, self.sigma, count)

    def probability(self, x: float, y: float) -> float:
        from scipy import stats
        return stats.norm(loc=self.mu, scale=self.sigma).cdf(y) - stats.norm(loc=self.mu, scale=self.sigma).cdf(x)

    def moment(self, k: int) -> float:
        if k < 0 or int(k) != k:
            raise ValueError("non-negative integer")
        k = int(k)
        if k == 0:
            return 1.0
        if k == 1:
            return self.mu
        if k == 2:
            return self.mu**2 + self.sigma**2
        return float(np.sum([math.comb(k, j) * self.mu**(k - j) * self.sigma**j * (0 if (k - j) % 2 == 1 else 1) for j in range(k + 1)]))

    def quantile(self, x: float) -> float:
        from scipy import stats
        return stats.norm(loc=self.mu, scale=self.sigma).ppf(x)

    def cdf(self, x: float) -> float:
        from scipy import stats
        return stats.norm(loc=self.mu, scale=self.sigma).cdf(x)

    @property
    def name(self) -> str:
        return "Normal" + "[" + super().name + "]" + "(" + str(self.mu) + "," + str(self.sigma) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.lognormal.exp import ExpNormalPattern
        from algebrandom.patterns.normal.addition import NormalAdditionPattern
        from algebrandom.patterns.normal.ratio import NormalRatioPattern
        from algebrandom.patterns.normal.scaling import NormalScalingPattern
        from algebrandom.patterns.normal.shift import NormalShiftPattern
        from algebrandom.patterns.normal.square import NormalSquarePattern
        from algebrandom.patterns.studentt.ratio import StudentTRatioPattern

        return [
            ExpNormalPattern(),
            NormalSquarePattern(),
            StudentTRatioPattern(),
            NormalRatioPattern(),
            NormalScalingPattern(),
            NormalShiftPattern(),
            NormalAdditionPattern(),
        ]


        
        
        
        