import numpy as np
from numpy.typing import NDArray
import sympy as sp

from algebrandom.core.primitive import RandomVariable


class PowRandomVariable(RandomVariable):
    
    def __init__(self,rv: RandomVariable, exponent: float, default_sample_count: int | None = None, rng: np.random.Generator | None = None) -> None:
            super().__init__(
                "(" + rv.name + ")^" + str(exponent),
                default_sample_count if default_sample_count is not None else rv._default_sample_count,
                rng if rng is not None else rv._rng,
                expression=rv.symbol ** exponent,
            )
            self._rv = rv
            self._exponent = exponent
            
    def _sample_no_cache(self, count: int | None = None, rng: np.random.Generator | None = None, cache : dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return self._rv.sample(count, rng, cache) ** self._exponent

    def probability(self, x: float, y: float) -> float:
        return self._rv.probability(x ** (1 / self._exponent), y ** (1 / self._exponent))

    def relevant_patterns(self):
        return self._rv.relevant_patterns()
