


from __future__ import annotations

from typing import TYPE_CHECKING
import sympy as sp

if TYPE_CHECKING:
    from algebrandom.core.primitive import RandomVariable


class MatchedPattern:
    
    pattern: sp.Expr
    match: dict[sp.Wild, sp.Basic]
    
    def __init__(self, pattern: sp.Expr, match: dict[sp.Wild, sp.Basic]):
        self.pattern = pattern
        self.match = match
        
    def get_rv[U : RandomVariable](self, wildcard: sp.Wild, type: type[U]) -> U:
        basic: sp.Basic = self.match[wildcard]
        rv = getattr(basic, "_rv", None)
        if rv is not None:
            return rv
        raise ValueError
        
    def get_constant(self, wildcard: sp.Wild) -> float:
        return float(self.match[wildcard]) # type: ignore
    
class Pattern():

    priority: int = 50
    
    def match(self, expr: sp.Expr) -> tuple[sp.Expr, "RandomVariable"] | None:
        ...
        
    def get_matched_pattern(self, expr: sp.Expr, pattern: sp.Expr) -> "MatchedPattern | None":
        match = expr.match(pattern)
        if match is not None:
            return MatchedPattern(expr, match)
        return None

    def wc_random_variable(self, name: str, rv_type: type[RandomVariable], positive: bool = False) -> sp.Wild:
        def has_exact_rv_type(expr: sp.Basic) -> bool:
            if getattr(expr, "_rv_type", None) is not rv_type:
                return False
            if positive:
                return expr.is_positive is True # type: ignore
            return True

        return sp.Wild(name, properties=[has_exact_rv_type])

    def wc_constant(self, name: str, positive: bool = False) -> sp.Wild:
        def is_constant(expr: sp.Basic) -> bool:
            if expr.is_number is not True:
                return False
            if positive:
                return expr.is_positive is True # type: ignore
            return True

        return sp.Wild(name, properties=[is_constant])
