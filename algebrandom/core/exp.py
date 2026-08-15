
import numpy as np
from numpy.typing import NDArray
import sympy as sp

from algebrandom.core.primitive import RandomVariable


class ExpRandomVariable(RandomVariable):
    
    def __init__(self,rv: RandomVariable, default_sample_count: int | None = None, rng: np.random.Generator | None = None, ) -> None:
        super().__init__(
            "exp(" + rv.name + ")",
            default_sample_count if default_sample_count is not None else rv._default_sample_count,
            rng if rng is not None else rv._rng,
            expression=sp.exp(rv.symbol), # type: ignore
        )
        self._rv = rv
        
    def _sample_no_cache(self, count: int | None = None, rng: np.random.Generator | None = None, cache : dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        return np.exp(self._rv.sample(count, rng, cache))
    
    def log(self) -> "RandomVariable":
        return self._rv
    
    def relevant_patterns(self):
        return self._rv.relevant_patterns()
