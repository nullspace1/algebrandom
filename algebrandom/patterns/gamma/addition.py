import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.gamma import Gamma, specialize_gamma


class GammaAdditionPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        result = self._match_gamma_gamma(expr)
        if result is not None:
            return result
        return self._match_gamma_exp(expr)

    def _match_gamma_gamma(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("left_gamma", Gamma)
        right_wildcard = self.wc_random_variable("right_gamma", Gamma)
        matched = self.get_matched_pattern(expr, left_wildcard + right_wildcard)
        if matched is None:
            return None
        left = matched.get_rv(left_wildcard, Gamma)
        right = matched.get_rv(right_wildcard, Gamma)
        if not left.is_independent_from(right):
            return None
        if not math.isclose(left.rate, right.rate):
            return None
        added = specialize_gamma(left.shape + right.shape, left.rate, left.name + "+" + right.name, expression=matched.pattern)
        return matched.pattern, added

    def _match_gamma_exp(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        gamma_wildcard = self.wc_random_variable("mix_gamma", Gamma)
        exp_wildcard = self.wc_random_variable("mix_exp", Exponential)
        matched = self.get_matched_pattern(expr, gamma_wildcard + exp_wildcard)
        if matched is None:
            return None
        gamma = matched.get_rv(gamma_wildcard, Gamma)
        exponential = matched.get_rv(exp_wildcard, Exponential)
        if not gamma.is_independent_from(exponential):
            return None
        if not math.isclose(gamma.rate, exponential.rate):
            return None
        added = specialize_gamma(gamma.shape + 1, gamma.rate, gamma.name + "+" + exponential.name, expression=matched.pattern)
        return matched.pattern, added
