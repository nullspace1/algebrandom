import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.chisquare import ChiSquare
from algebrandom.instances.gamma import specialize_gamma
from algebrandom.instances.normal import Normal


class NormalSquarePattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        rv_wildcard = self.wc_random_variable("squared_normal", Normal)
        pattern = rv_wildcard ** 2

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        rv = matched.get_rv(rv_wildcard, Normal)
        if rv.mu != 0:
            return None

        if math.isclose(rv.sigma, 1.0):
            reduced = ChiSquare(1, rv.name + "^2", expression=matched.pattern)
        else:
            reduced = specialize_gamma(0.5, 1 / (2 * rv.sigma**2), rv.name + "^2", expression=matched.pattern)
        return matched.pattern, reduced
