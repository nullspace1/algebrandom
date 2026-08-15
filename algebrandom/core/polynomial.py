from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import sympy as sp

from algebrandom.core.primitive import RandomVariable



class PolynomialRandomVariable(RandomVariable):
    
    __MAX_NAME_LENGTH = 10
    
    _rvs: set[RandomVariable]

    @property
    def name(self) -> str:
        if len(self._rvs) > self.__MAX_NAME_LENGTH:
            return f"P(... {len(self._rvs)} ...)"
        return "P(" + ",".join(rv.name for rv in self._rvs) + ")"

    @property
    def operands(self) -> set[RandomVariable]:
        return set(self._rvs)

    def get_rv(self, symbol : sp.Symbol ) -> RandomVariable | None:
        for rv in self._rvs:
            if rv.symbol == symbol:
                return rv
        raise KeyError
    

    def __init__(
        self,
        polynomial: sp.Expr,
        name: str = "",
        default_sample_count: int = 100,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rvs = RandomVariable._rvs_from_expr(polynomial)
        super().__init__(name, default_sample_count, rng, expression=polynomial)

    def _sample_no_cache(
        self,
        count: int | None = None,
        rng: np.random.Generator | None = None,
        cache: dict[RandomVariable, NDArray[np.float64]] = {},
    ) -> NDArray[np.float64]:
        ordered = tuple(self._rvs)
        function = sp.lambdify([rv.symbol for rv in ordered], self._expression, modules="numpy")
        values = (rv.sample(count, rng, cache) for rv in ordered)
        return function(*values)

    def _symbol_rv(self, expr: sp.Expr) -> RandomVariable | None:
        if expr.is_symbol:
            return getattr(expr, "_rv", None)
        return None

    def _reverse_sub(self, expr: sp.Expr, composites: list[tuple[sp.Expr, RandomVariable]]) -> sp.Expr:
        from algebrandom.core.exp import ExpRandomVariable
        from algebrandom.core.log import LogRandomVariable
        from algebrandom.core.pow import PowRandomVariable

        wrappers = (LogRandomVariable, ExpRandomVariable, PowRandomVariable, PolynomialRandomVariable)
        rewritten = expr
        stored = set(self._rvs)
        for atomic_form, rv in sorted(composites, key=lambda item: sp.count_ops(item[0]), reverse=True):
            if rv not in stored or isinstance(rv, wrappers):
                continue
            rewritten = rewritten.subs(atomic_form, rv.symbol)
        return rewritten # type: ignore

    def simplify(self) -> RandomVariable:
        atomic, composites = self._unfold(self._expression)
        cancelled = sp.cancel(sp.simplify(atomic))
        rv = self._symbol_rv(cancelled)
        if rv is not None:
            return rv

        rewritten = self._reverse_sub(cancelled, composites)
        rv = self._symbol_rv(rewritten)
        if rv is not None:
            return rv

        while True:
            rvs = self._rvs_from_expr(rewritten)
            matched_any = False
            for pattern in self._patterns_for(rvs):
                match = pattern.match(rewritten)
                if match is None:
                    continue
                matched_expr, reduced = match
                new_expr: sp.Expr = rewritten.subs(matched_expr, reduced.symbol) # type: ignore
                if new_expr.equals(rewritten):
                    continue
                rewritten = new_expr
                matched_any = True
                rv = self._symbol_rv(rewritten)
                if rv is not None:
                    return rv
                break
            if not matched_any:
                break

        if rewritten.equals(self._expression):
            return self
        return PolynomialRandomVariable(rewritten, self._name, self._default_sample_count, self._rng)

    def _patterns_for(self, rvs: set[RandomVariable]) -> list:
        from algebrandom.core.pattern import Pattern

        merged: dict[type, Pattern] = {}
        for rv in rvs:
            for pattern in rv.relevant_patterns():
                merged.setdefault(type(pattern), pattern)
        return sorted(merged.values(), key=lambda pattern: (pattern.priority, type(pattern).__name__))

    def relevant_patterns(self) -> list:
        return self._patterns_for(self._rvs)
