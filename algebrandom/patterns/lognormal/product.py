import sympy as sp

from algebrandom.core.pattern import Pattern
from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.lognormal import LogNormal


class LogNormalProductPattern(Pattern):
    priority = 40

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        left_wildcard = self.wc_random_variable("left_lognormal", LogNormal)
        right_wildcard = self.wc_random_variable("right_lognormal", LogNormal)
        pattern = left_wildcard * right_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        left = matched.get_rv(left_wildcard, LogNormal)
        right = matched.get_rv(right_wildcard, LogNormal)
        if not left.is_independent_from(right):
            return None

        product = LogNormal(
            left.mu + right.mu,
            (left.sigma ** 2 + right.sigma ** 2) ** 0.5,
            left.name + "*" + right.name,
            expression=matched.pattern,
        )
        return matched.pattern, product
