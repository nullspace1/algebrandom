import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.cauchy import Cauchy
from algebrandom.instances.normal import Normal


class NormalRatioPattern(Pattern):
    priority = 20

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("num_normal", Normal)
        right_wildcard = self.wc_random_variable("den_normal", Normal)
        pattern = left_wildcard / right_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        left = matched.get_rv(left_wildcard, Normal)
        right = matched.get_rv(right_wildcard, Normal)
        if left.mu != 0 or right.mu != 0:
            return None
        if not left.is_independent_from(right):
            return None

        reduced = Cauchy(0.0, left.sigma / right.sigma, left.name + "/" + right.name, expression=matched.pattern)
        return matched.pattern, reduced
