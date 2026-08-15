import numpy as np
from numpy.typing import NDArray
from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
import sympy as sp



class Uniform(RandomVariable):
    
    min: float
    max: float
    
    def __init__(self, min: float, max: float, name: str, default_sample_count: int = 100, rng: np.random.Generator | None = None, **kwargs) -> None:
        self.min = min
        self.max = max
        super().__init__(name, default_sample_count, rng, **kwargs)
        
    def _sample_no_cache(self, count: int | None = None, rng: np.random.Generator | None = None, cache : dict["RandomVariable", NDArray[np.float64]] = {}) -> np.ndarray:
        return np.array(self._rng.uniform(self.min, self.max, count))
    
    def probability(self, x: float, y: float) -> float:
        return (y - x) / (self.max - self.min)
    
    def moment(self, k: int) -> float:
        return np.power(self.max - self.min, k)
    
    def quantile(self, x: float) -> float:
        if x < 0:
            return 0
        if x > 1:
            return 1
        return self.min + x * (self.max - self.min)
    
    def cdf(self, x: float) -> float:
        return (x - self.min) / (self.max - self.min)
    
    @property
    def name(self) -> str:
        name = super().name
        return "Uniform" + "[" +  name + "]" + "("+ str(self.min) + "," + str(self.max) + ")"

    @classmethod
    def _build_patterns(cls):
        from algebrandom.patterns.pareto.transforms import UniformToParetoPattern
        from algebrandom.patterns.uniform.log import LogUniformPattern
        from algebrandom.patterns.uniform.scaling import UniformScalingPattern
        from algebrandom.patterns.uniform.shift import UniformShiftPattern

        return [
            LogUniformPattern(),
            UniformToParetoPattern(),
            UniformScalingPattern(),
            UniformShiftPattern(),
        ]

    
    

