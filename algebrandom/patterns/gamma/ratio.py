import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.beta import specialize_beta
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.gamma import Gamma


class GammaRatioPattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        from algebrandom.instances.chisquare import ChiSquare

        result = self._match_pair(expr, ChiSquare, ChiSquare, lambda a, b: (a.df / 2, b.df / 2, 0.5, 0.5, a, b))
        if result is not None:
            return result
        result = self._match_pair(expr, Gamma, Gamma, lambda a, b: (a.shape, b.shape, a.rate, b.rate, a, b))
        if result is not None:
            return result
        result = self._match_pair(expr, Exponential, Exponential, lambda a, b: (1.0, 1.0, a.rate, b.rate, a, b))
        if result is not None:
            return result
        result = self._match_pair(expr, Gamma, Exponential, lambda a, b: (a.shape, 1.0, a.rate, b.rate, a, b))
        if result is not None:
            return result
        result = self._match_pair(expr, Exponential, Gamma, lambda a, b: (1.0, b.shape, a.rate, b.rate, a, b))
        if result is not None:
            return result
        return self._match_over_sum(expr)

    def _match_over_sum(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("sum_num", Gamma)
        right_wildcard = self.wc_random_variable("sum_den", Gamma)
        matched = self.get_matched_pattern(expr, left_wildcard / right_wildcard)
        if matched is None:
            left_wildcard = self.wc_random_variable("sum_num_e", Exponential)
            matched = self.get_matched_pattern(expr, left_wildcard / right_wildcard)
            if matched is None:
                return None
            left = matched.get_rv(left_wildcard, Exponential)
            right = matched.get_rv(right_wildcard, Gamma)
            left_shape, left_rate = 1.0, left.rate
        else:
            left = matched.get_rv(left_wildcard, Gamma)
            right = matched.get_rv(right_wildcard, Gamma)
            left_shape, left_rate = left.shape, left.rate
        if left not in right.dependencies:
            return None
        if not math.isclose(left_rate, right.rate):
            return None
        rest = right.shape - left_shape
        if rest <= 0:
            return None
        reduced = specialize_beta(left_shape, rest, left.name + "/" + right.name, expression=matched.pattern)
        return matched.pattern, reduced

    def _match_pair(self, expr: sp.Expr, left_type: type, right_type: type, params) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("ratio_left", left_type)
        right_wildcard = self.wc_random_variable("ratio_right", right_type)
        matched = self.get_matched_pattern(expr, left_wildcard / (left_wildcard + right_wildcard))
        if matched is None:
            return None
        left = matched.get_rv(left_wildcard, left_type)
        right = matched.get_rv(right_wildcard, right_type)
        if not left.is_independent_from(right):
            return None
        alpha, beta, left_rate, right_rate, left_rv, right_rv = params(left, right)
        if not math.isclose(left_rate, right_rate):
            return None
        reduced = specialize_beta(alpha, beta, left_rv.name + "/(" + left_rv.name + "+" + right_rv.name + ")", expression=matched.pattern)
        return matched.pattern, reduced
