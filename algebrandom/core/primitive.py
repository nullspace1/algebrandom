from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol

if TYPE_CHECKING:
    from algebrandom.core.pattern import Pattern
    from algebrandom.core.exp import ExpRandomVariable
    from algebrandom.core.polynomial import PolynomialRandomVariable
    from algebrandom.core.pow import PowRandomVariable

from algebrandom.core.symbol import RandomVariableSymbol

class RandomVariable:
    
    _symbol : Symbol
    _default_sample_count: int
    _rng : np.random.Generator
    _name: str
    _expression: Expr
    
    
    def __init__(self, name: str, default_sample_count: int = 1000, rng: np.random.Generator | None = None, *, expression: Expr | None = None) -> None:
        self._name = name
        self._default_sample_count = default_sample_count
        self._rng = np.random.default_rng() if rng is None else rng
        self.type = type(self)
        self._symbol = RandomVariableSymbol(self)
        self._expression = self._symbol if expression is None else expression
        
    
    @property
    def name(self) -> str:
        return self._name
    
    def rename(self, name: str) -> "RandomVariable":
        self._name = name
        return self
    
    
    @property
    def dependencies(self) -> set["RandomVariable"]:
        return self._rvs_from_expr(self.expr())
    
    def __str__(self) -> str:
        return self.name
    
    def expr(self) -> Expr:
        atomic, _ = self._unfold(self._expression)
        return atomic

    def _unfold(self, expr: Expr) -> tuple[Expr, list[tuple[Expr, "RandomVariable"]]]:
        composites: list[tuple[Expr, RandomVariable]] = []
        unfolded: dict[int, Expr] = {}

        def unfold_rv(rv: RandomVariable) -> Expr:
            cached = unfolded.get(id(rv))
            if cached is not None:
                return cached
            if rv._expression == rv.symbol:
                unfolded[id(rv)] = rv.symbol
                return rv.symbol
            atomic = unfold_symbols(rv._expression)
            unfolded[id(rv)] = atomic
            composites.append((atomic, rv))
            return atomic

        def unfold_symbols(e: Expr) -> Expr:
            result = e
            for symbol in list(e.free_symbols):
                rv = getattr(symbol, "_rv", None)
                if rv is None:
                    continue
                result = result.subs(symbol, unfold_rv(rv))
            return result # type: ignore

        return unfold_symbols(expr), composites
    
    @property
    def symbol(self) -> Symbol:
        return self._symbol
    
    def is_independent_from(self, rv: "RandomVariable") -> bool:
        return self.dependencies.isdisjoint(rv.dependencies)

    @classmethod
    def _build_patterns(cls) -> list["Pattern"]:
        return []

    @classmethod
    def pattern_set(cls) -> list["Pattern"]:
        cached = cls.__dict__.get("_pattern_cache")
        if cached is None:
            cached = sorted(cls._build_patterns(), key=lambda pattern: (pattern.priority, type(pattern).__name__))
            cls._pattern_cache = cached
        return cached

    def relevant_patterns(self) -> list["Pattern"]:
        return list(type(self).pattern_set())

    def simplify(self) -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable

        expr = self._expression
        for pattern in self.relevant_patterns():
            match = pattern.match(expr)
            if match is None:
                continue
            matched_expr, rv = match
            new_expr : Expr = expr.subs(matched_expr, rv.symbol) # type: ignore
            if new_expr.equals(rv.symbol): # type: ignore
                return rv
            return PolynomialRandomVariable(new_expr, self._name, self._default_sample_count, self._rng).simplify()
        return self

    def _sample_no_cache(self, count: int, rng: np.random.Generator, cache : dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        ...
        
    def sample(self, count: int | None = None, rng: np.random.Generator | None = None, cache : dict["RandomVariable", NDArray[np.float64]] = {}) -> NDArray[np.float64]:
        if self in cache:
            return cache[self]
        count = self._default_sample_count if count is None else count
        rng = self._rng if rng is None else rng
        sample = self._sample_no_cache(count, rng, cache)
        cache[self] = sample
        return sample
        
    def probability(self, x: float, y: float) -> float:
        return np.mean(np.logical_and(self.sample() > x, self.sample() < y))
        
    def moment(self, k: int) -> float:
        return np.mean(np.power(self.sample(),k))
        
    def quantile(self, x: float) -> float:
        return np.quantile(self.sample(), x)
        
    def cdf(self, x: float) -> float:
        return self.probability(-np.inf, x)
    
    def mean(self) -> float:
        return self.moment(1)
    
    def variance(self) -> float:
        return self.moment(2) - self.moment(1)**2
    
    def stdev(self) -> float:
        return np.sqrt(self.variance())
    
    def log(self) -> "RandomVariable":
        from algebrandom.core.log import LogRandomVariable
        return LogRandomVariable(self).simplify()

    def exp(self) -> "RandomVariable":
        from algebrandom.core.exp import ExpRandomVariable
        return ExpRandomVariable(self).simplify()

    def __neg__(self) -> "RandomVariable":
        return self * -1

    def __pow__(self,exponent: "int | float | RandomVariable") -> "RandomVariable":
        if isinstance(exponent, int):
            from algebrandom.core.polynomial import PolynomialRandomVariable
            return PolynomialRandomVariable(self.symbol ** exponent).simplify()
        if isinstance(exponent, float):
            from algebrandom.core.pow import PowRandomVariable
            return PowRandomVariable(self, exponent).simplify()

        from algebrandom.core.exp import ExpRandomVariable
        return ExpRandomVariable(rv=self.log() * exponent).simplify()
    
    def __mul__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable    
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(self.symbol * val.symbol).simplify()
        return PolynomialRandomVariable(self.symbol * val).simplify()
    
    def __add__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(self.symbol + val.symbol).simplify()
        return PolynomialRandomVariable(self.symbol + val).simplify()
    
    def __sub__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(self.symbol - val.symbol).simplify()
        return PolynomialRandomVariable(self.symbol - val).simplify()
    
    def __truediv__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(self.symbol / val.symbol).simplify()
        return PolynomialRandomVariable(self.symbol / val).simplify()

    def __rpow__(self, base: "int | float | RandomVariable") -> "RandomVariable":
        from algebrandom.core.exp import ExpRandomVariable
        if isinstance(base, (int, float)):
            import math
            return ExpRandomVariable(rv=self * math.log(base)).simplify()
        if isinstance(base, RandomVariable):
            return base.__pow__(self)
        raise NotImplementedError

    def __rmul__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(val.symbol * self.symbol).simplify()
        return PolynomialRandomVariable(val * self.symbol).simplify()

    def __radd__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(val.symbol + self.symbol).simplify()
        return PolynomialRandomVariable(val + self.symbol).simplify()

    def __rsub__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(val.symbol - self.symbol).simplify()
        return PolynomialRandomVariable(val - self.symbol).simplify()

    def __rtruediv__(self, val: "RandomVariable | float") -> "RandomVariable":
        from algebrandom.core.polynomial import PolynomialRandomVariable
        if isinstance(val, RandomVariable):
            return PolynomialRandomVariable(val.symbol / self.symbol).simplify()
        return PolynomialRandomVariable(val / self.symbol).simplify()


    @staticmethod
    def _rvs_from_expr(expr: Expr) -> set[RandomVariable]:
        result: set[RandomVariable] = set()
        for symbol in expr.free_symbols:
            rv = getattr(symbol, "_rv", None)
            if rv is not None:
                result.add(rv)
        return result
