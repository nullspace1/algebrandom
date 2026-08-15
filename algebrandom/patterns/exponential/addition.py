import math

import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.exponential import Exponential
from algebrandom.instances.gamma import specialize_gamma


class ExponentialAdditionPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("left_exp", Exponential)
        right_wildcard = self.wc_random_variable("right_exp", Exponential)
        pattern = left_wildcard + right_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        left = matched.get_rv(left_wildcard, Exponential)
        right = matched.get_rv(right_wildcard, Exponential)
        if not left.is_independent_from(right):
            return None
        if not math.isclose(left.rate, right.rate):
            return None

        added = specialize_gamma(2.0, left.rate, left.name + "+" + right.name, expression=matched.pattern)
        return matched.pattern, added
